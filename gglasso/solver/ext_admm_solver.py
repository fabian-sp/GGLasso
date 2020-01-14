"""
author: Fabian Schaipp
"""

import numpy as np
import time
import copy

from .ggl_helper import phiplus, prox_od_1norm


def ext_ADMM_MGL(S, lambda1, lambda2, reg , Omega_0, G,\
             eps_admm = 1e-5 , verbose = False, measure = False, **kwargs):
    """
    This is an ADMM algorithm for solving the Multiple Graphical Lasso problem
    where not all instances have the same number of dimensions
    reg specifies the type of penalty, i.e. Group or Fused Graphical Lasso
    
    Omega_0: start point -- must be specified as a dictionary with the keys 0,...,K-1 (as integers)
    S: empirical covariance matrices -- must be specified as a dictionary with the keys 0,...,K-1 (as integers)
    G: array containing the group penalty indices
    max_iter and rho can be specified via kwargs
    """

    assert reg in ['GGL']
    assert min(lambda1, lambda2) > 0
        
    K = len(S.keys())
    p = np.zeros(K, dtype= int)
    for k in np.arange(K):
        p[k] = S[k].shape[0]
        
    
    if 'max_iter' in kwargs.keys():
        max_iter = kwargs.get('max_iter')
    else:
        max_iter = 1000
    if 'rho' in kwargs.keys():
        assert kwargs.get('rho') > 0
        rho = kwargs.get('rho')
    else:
        rho = 1.
        
    
    # initialize 
    status = 'not optimal'
    Omega_t = Omega_0.copy()
    Theta_t = Omega_0.copy()
    Lambda_t = Omega_0.copy()
    
    Z_t = dict()
    X0_t = dict()
    X1_t = dict()
    for k in np.arange(K):
        X0_t[k] = np.zeros((p[k],p[k]))
        X1_t[k] = np.zeros((p[k],p[k]))
     
     
    runtime = np.zeros(max_iter)
    kkt_residual = np.zeros(max_iter)
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
            
        eta_A = ext_ADMM_stopping_criterion(Omega_t, Theta_t, Lambda_t, X0_t, X1_t, S , G, lambda1, lambda2, reg)
        kkt_residual[iter_t] = eta_A
            
        if eta_A <= eps_admm:
            status = 'optimal'
            break
        if verbose:
            print(f"------------Iteration {iter_t} of the ADMM Algorithm----------------")
        
        # Omega Update
        for k in np.arange(K):
            W_t = Theta_t[k] - X0_t[k] - (1/rho) * S[k]
            eigD, eigQ = np.linalg.eigh(W_t)
            Omega_t[k] = phiplus(W_t, beta = 1/rho, D = eigD, Q = eigQ)
        
        # Theta Update
        for k in np.arange(K): 
            V_t = (Omega_t[k] + X0_t[k] + Lambda_t[k] - X1_t[k]) * 0.5
            Theta_t[k] = prox_od_1norm(V_t, lambda1/(2*rho))
        
        # Lambda Update
        for k in np.arange(K): 
            Z_t[k] = Theta_t[k] + X1_t[k]
            
        Lambda_t = prox_2norm_G(Z_t, G, lambda2/rho)
        
        # X Update
        for k in np.arange(K):
            X0_t[k] +=  Omega_t[k] - Theta_t[k]
            X1_t[k] +=  Theta_t[k] - Lambda_t[k]
        
        if measure:
            end = time.time()
            runtime[iter_t] = end-start
            
        if verbose:
            print(f"Current accuracy: ", eta_A)
        
    if eta_A > eps_admm:
        status = 'max iterations reached'
        
    print(f"ADMM terminated after {iter_t} iterations with accuracy {eta_A}")
    print(f"ADMM status: {status}")
    
    for k in np.arange(K):
        assert abs(Omega_t[k].T - Omega_t[k]).max() <= 1e-5, "Solution is not symmetric"
        assert abs(Theta_t[k].T - Theta_t[k]).max() <= 1e-5, "Solution is not symmetric"
    
    sol = {'Omega': Omega_t, 'Theta': Theta_t, 'X0': X0_t, 'X1': X1_t}
    if measure:
        info = {'status': status , 'runtime': runtime[:iter_t], 'kkt_residual': kkt_residual[1:iter_t +1]}
    else:
        info = {'status': status}
               
    return sol, info

def ext_ADMM_stopping_criterion(Omega, Theta, Lambda, X0, X1, S , G, lambda1, lambda2, reg):
    
    K = len(S.keys())
    term1 = np.zeros(K)
    term2 = np.zeros(K)
    term3 = np.zeros(K)
    term4 = np.zeros(K)
    term5 = np.zeros(K)
    V = dict()
    
    for k in np.arange(K):
        eigD, eigQ = np.linalg.eigh(Omega[k] - S[k] - X0[k])
        proxk = phiplus(Omega[k] - S[k] - X0[k], beta = 1, D = eigD, Q = eigQ)
        term1[k] = np.linalg.norm(Omega[k] - proxk) / (1 + np.linalg.norm(Omega[k]))
        
        term2[k] = np.linalg.norm(Theta[k] - prox_od_1norm(Theta[k] + X0[k] - X1[k] , lambda1)) / (1 + np.linalg.norm(Theta[k]))
        
        V[k] = Lambda[k] + X1[k]
       
        term4[k] = np.linalg.norm(Omega[k] - Theta[k]) / (1 + np.linalg.norm(Theta[k]))
        term5[k] = np.linalg.norm(Lambda[k] - Theta[k]) / (1 + np.linalg.norm(Theta[k]))
    
    
    V = prox_2norm_G(V, G, lambda2)
    for k in np.arange(K):
        term3[k] = np.linalg.norm(V[k] - Lambda[k]) / (1 + np.linalg.norm(Lambda[k]))
    
    res = max(np.linalg.norm(term1), np.linalg.norm(term2), np.linalg.norm(term3), np.linalg.norm(term4), np.linalg.norm(term5) )
    return res


def construct_G(p, K):
    L = int(p*(p-1)/2)
    G = np.zeros((2,L,K), dtype = int)
    for i in np.arange(p):
        for j in np.arange(start = i+1, stop =p):       
            ix = lambda i,j : i*p - int(i*(i+1)/2) + j - 1 - i*1
            #print(ix(i,j))
            G[0, ix(i,j), :] = i
            G[1, ix(i,j), :] = j
    return G

def check_G(G, p):
    """
    function to check a bookkeeping group penalty matrix G
    p: vector of length K with dimensions p_k as entries
    """
    
    assert G.dtype == int, "G needs to be an integer array"
    
    assert np.all(G[0,:,:] != G[1,:,:]), "G has entries on the diagonal!"
    
    assert np.all(G >=0), "No negative indices allowed"
    
    assert np.all(G.max(axis = (0,1)) < p), "indices larger as dimension were found"
    
    return


def prox_2norm_G(X, G, l2):
    """
    calculates the proximal operator at points X for the group penalty induced by G
    G: 2xLxK matrix where the -th row contains the (i,j)-index of the element in Theta^k which contains to group l
       if G has a np.nan entry no element is contained in the group for this Theta^k
    X: dictionary with X^k at key k, each X[k] is assumed to be symmetric
    """
    assert l2 > 0
    K = len(X.keys())
    for  k in np.arange(K):
        assert abs(X[k] - X[k].T).max() <= 1e-5, "X[k] has to be symmteric"
    
    d = G.shape
    assert d[0] == 2
    assert d[2] == K
    L = d[1]
    
    X1 = copy.deepcopy(X)
    
    for l in np.arange(L):
        # for each group construct v, calculate prox, and insert the result. Ignore NaN entries of G
        v0 = np.zeros(K)
        for k in np.arange(K):
            if G[0,l,k] == np.nan:
                v0[k] = 0
            else:
                v0[k] = X[k][G[0,l,k], G[1,l,k]]
        
        #v = v0[~np.isnan(v0)]
        a = max(np.linalg.norm(v0,2) , l2)
        z = v0 * (a - l2) / a
        
        for k in np.arange(K):
            if G[0,l,k] == np.nan:
                continue
            else:
                X1[k][G[0,l,k], G[1,l,k]] = z[k]
                # lower triangular
                X1[k][G[1,l,k], G[0,l,k]] = z[k]
             
    return X1

    
########################################################################


#l = .1
#X = {}
#refX = np.zeros((K,p,p))
#for k in np.arange(K):
#    A = np.random.randn(p,p)
#    X[k] = A.T @ A
#    refX[k,:,:] = X[k].copy()
#    
#    
#X1 = prox_2norm_G(X, G, l)
#
#X1arr = np.zeros((K,p,p))
#for k in np.arange(K):
#    X1arr[k,:,:] = X1[k].copy()
#
#refY = prox_chi(refX, l)    
#
#np.linalg.norm(refY-X1arr)


    