"""
author: Fabian Schaipp
"""

import numpy as np
import time
import copy

from .ggl_helper import phiplus, prox_od_1norm, prox_2norm, prox_rank_norm
from ..helper.ext_admm_helper import check_G


def ext_ADMM_MGL(S, lambda1, lambda2, reg , Omega_0, G,\
             X0 = None, X1 = None, eps_admm = 1e-5 , verbose = False, measure = False, latent = False, mu1 = None, **kwargs):
    """
    This is an ADMM algorithm for solving the Multiple Graphical Lasso problem
    where not all instances have the same number of dimensions
    reg specifies the type of penalty, i.e. Group or Fused Graphical Lasso
    
    Omega_0: start point -- must be specified as a dictionary with the keys 0,...,K-1 (as integers)
    S: empirical covariance matrices -- must be specified as a dictionary with the keys 0,...,K-1 (as integers)
    lambda1: can be a vector of length K or a float
    
    latent: boolean to indidate whether low rank term should be estimated
    mu1: low rank penalty parameter (if latent=True), can be a vector of length K or a float
    
    G: array containing the group penalty indices
    max_iter and rho can be specified via kwargs
    
    In the code, X are the SCALED dual variables, for the KKT stop criterion they have to be unscaled again!
    """
    K = len(S.keys())
    p = np.zeros(K, dtype= int)
    for k in np.arange(K):
        p[k] = S[k].shape[0]
        
    if type(lambda1) == np.float64 or type(lambda1) == float:
        lambda1 = lambda1*np.ones(K)
    if latent:
        if type(mu1) == np.float64 or type(mu1) == float:
             mu1 = mu1*np.ones(K)
            
        assert mu1 is not None
        assert np.all(mu1 > 0)
        
    assert min(lambda1.min(), lambda2) > 0
    assert reg in ['GGL']
   
    check_G(G, p)
    
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
    L_t = dict()
    Lambda_t = Omega_0.copy()
    
    for k in np.arange(K):
        L_t[k] = np.zeros((p[k],p[k]))
    
    # helper and dual variables
    Z_t = dict()

    if X0 == None:
        X0_t = dict()
        for k in np.arange(K):
            X0_t[k] = np.zeros((p[k],p[k]))  
    else:
        X0_t = X0.copy()
        
    if X1 == None:   
        X1_t = dict()
        for k in np.arange(K):
            X1_t[k] = np.zeros((p[k],p[k]))
    else:
        X1_t = X1.copy()
     
     
    runtime = np.zeros(max_iter)
    kkt_residual = np.zeros(max_iter)
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
        
        if iter_t % 10 == 0:
            eta_A = ext_ADMM_stopping_criterion(Omega_t, Theta_t, L_t, Lambda_t, dict((k, rho*v) for k,v in X0_t.items()), dict((k, rho*v) for k,v in X1_t.items()),\
                                                S , G, lambda1, lambda2, reg, latent, mu1)
            kkt_residual[iter_t] = eta_A
            
        if eta_A <= eps_admm:
            status = 'optimal'
            break
        if verbose:
            print(f"------------Iteration {iter_t} of the ADMM Algorithm----------------")
        
        # Omega Update
        for k in np.arange(K):
            W_t = Theta_t[k] - L_t[k] - X0_t[k] - (1/rho) * S[k]
            eigD, eigQ = np.linalg.eigh(W_t)
            Omega_t[k] = phiplus(W_t, beta = 1/rho, D = eigD, Q = eigQ)
        
        # Theta Update
        for k in np.arange(K): 
            V_t = (Omega_t[k] + L_t[k] + X0_t[k] + Lambda_t[k] - X1_t[k]) * 0.5
            Theta_t[k] = prox_od_1norm(V_t, lambda1[k]/(2*rho))
        
        #L Update
        if latent:
            for k in np.arange(K):
                C_t = Theta_t[k] - X0_t[k] - Omega_t[k]
                C_t = (C_t.T + C_t)/2
                eigD, eigQ = np.linalg.eigh(C_t)
                L_t[k] = prox_rank_norm(C_t, mu1[k]/rho, D = eigD, Q = eigQ)
        
        # Lambda Update
        for k in np.arange(K): 
            Z_t[k] = Theta_t[k] + X1_t[k]
            
        Lambda_t = prox_2norm_G(Z_t, G, lambda2/rho)
        # X Update
        for k in np.arange(K):
            X0_t[k] +=  Omega_t[k] - Theta_t[k] + L_t[k]
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
        assert abs(L_t[k].T - L_t[k]).max() <= 1e-5, "Solution is not symmetric"
        
        D,_ = np.linalg.eigh(Theta_t[k])
        if D.min() <= 1e-5:
            print("WARNING: Theta may be not positive definite -- increase accuracy!")
                     
        D,_ = np.linalg.eigh(L_t[k])
        if D.min() <= -1e-5:
            print("WARNING: L may be not positive semidefinite -- increase accuracy!")
    
    
    sol = {'Omega': Omega_t, 'Theta': Theta_t, 'L': L_t, 'X0': X0_t, 'X1': X1_t}
    if measure:
        info = {'status': status , 'runtime': runtime[:iter_t], 'kkt_residual': kkt_residual[1:iter_t +1]}
    else:
        info = {'status': status}
               
    return sol, info

def ext_ADMM_stopping_criterion(Omega, Theta, L, Lambda, X0, X1, S , G, lambda1, lambda2, reg, latent = False, mu1 = None):
    
    K = len(S.keys())
    
    if not latent:
        for k in np.arange(K):
            assert np.all(L[k]==0)
        
    term1 = np.zeros(K)
    term2 = np.zeros(K)
    term3 = np.zeros(K)
    term4 = np.zeros(K)
    term5 = np.zeros(K)
    term6 = np.zeros(K)
    V = dict()
    
    for k in np.arange(K):
        eigD, eigQ = np.linalg.eigh(Omega[k] - S[k] - X0[k])
        proxk = phiplus(Omega[k] - S[k] - X0[k], beta = 1, D = eigD, Q = eigQ)
        # primal varibale optimality
        term1[k] = np.linalg.norm(Omega[k] - proxk) / (1 + np.linalg.norm(Omega[k]))
        term2[k] = np.linalg.norm(Theta[k] - prox_od_1norm(Theta[k] + X0[k] - X1[k] , lambda1[k])) / (1 + np.linalg.norm(Theta[k]))
        
        if latent:
            eigD, eigQ = np.linalg.eigh(L[k] - X0[k])
            proxk = prox_rank_norm(L[k] - X0[k], beta = mu1[k], D = eigD, Q = eigQ)
            term3[k] = np.linalg.norm(L[k] - proxk) / (1 + np.linalg.norm(L[k]))
        
        V[k] = Lambda[k] + X1[k]
        
        # equality constraints
        term5[k] = np.linalg.norm(Omega[k] - Theta[k] + L[k]) / (1 + np.linalg.norm(Theta[k]))
        term6[k] = np.linalg.norm(Lambda[k] - Theta[k]) / (1 + np.linalg.norm(Theta[k]))
    
    
    V = prox_2norm_G(V, G, lambda2)
    for k in np.arange(K):
        term4[k] = np.linalg.norm(V[k] - Lambda[k]) / (1 + np.linalg.norm(Lambda[k]))
    
    res = max(np.linalg.norm(term1), np.linalg.norm(term2), np.linalg.norm(term3), np.linalg.norm(term4), np.linalg.norm(term5), np.linalg.norm(term6) )
    return res


def prox_2norm_G(X, G, l2):
    """
    calculates the proximal operator at points X for the group penalty induced by G
    G: 2xLxK matrix where the -th row contains the (i,j)-index of the element in Theta^k which contains to group l
       if G has a entry -1 no element is contained in the group for this Theta^k
    X: dictionary with X^k at key k, each X[k] is assumed to be symmetric
    """
    assert l2 > 0
    K = len(X.keys())
    for  k in np.arange(K):
        assert abs(X[k] - X[k].T).max() <= 1e-5, "X[k] has to be symmetric"
    
    d = G.shape
    assert d[0] == 2
    assert d[2] == K
    L = d[1]
    
    X1 = copy.deepcopy(X)
    group_size = (G[0,:,:] != -1).sum(axis = 1)
    
    for l in np.arange(L):
        # for each group construct v, calculate prox, and insert the result. Ignore -1 entries of G
        v0 = np.zeros(K)
        for k in np.arange(K):
            if G[0,l,k] == -1:
                v0[k] = np.nan
            else:
                v0[k] = X[k][G[0,l,k], G[1,l,k]]
        
        v = v0[~np.isnan(v0)]
        # scale with square root of the group size
        z0 = prox_2norm(v,l2 * np.sqrt(group_size[l]))
        v0[~np.isnan(v0)] = z0
        
        for k in np.arange(K):
            if G[0,l,k] == -1:
                continue
            else:
                X1[k][G[0,l,k], G[1,l,k]] = v0[k]
                # lower triangular
                X1[k][G[1,l,k], G[0,l,k]] = v0[k]
             
    return X1

    


    