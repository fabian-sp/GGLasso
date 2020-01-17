"""
author: Fabian Schaipp
"""

import numpy as np
import time

from .ggl_helper import phiplus, prox_od_1norm, prox_2norm, prox_rank_norm
from .ext_admm_solver import prox_2norm_G
from ..helper.ext_admm_helper import check_G


def ext_ADMM_MGL(S, lambda1, lambda2, mu1, reg , R_0, G,\
             eps_admm = 1e-5 , verbose = False, measure = False, **kwargs):
    """
    This is an ADMM algorithm for solving the Multiple Graphical Lasso problem with Latent Varibales
    where not all instances have the same number of dimensions
    reg specifies the type of penalty, i.e. Group or Fused Graphical Lasso
    
    R_0: start point -- must be specified as a dictionary with the keys 0,...,K-1 (as integers)
    S: empirical covariance matrices -- must be specified as a dictionary with the keys 0,...,K-1 (as integers)
    lambda1: can be a vector of length K or a float
    mu1: can be a vector of length K or a float
    G: array containing the group penalty indices
    max_iter and rho can be specified via kwargs
    """
    K = len(S.keys())
    p = np.zeros(K, dtype= int)
    for k in np.arange(K):
        p[k] = S[k].shape[0]
        
    if type(lambda1) == np.float64 or type(lambda1) == float:
        lambda1 = lambda1*np.ones(K)
    
    if type(mu1) == np.float64 or type(mu1) == float:
        mu1 = mu1*np.ones(K)
        
    assert min(lambda1.min(), lambda2) > 0
    assert mu1.min() > 0
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
    R_t = R_0.copy()
    Theta_t = R_0.copy()
    Lambda_t = R_0.copy()
    
    L_t = dict()
    # helper variable
    Z_t = dict()
    X0_t = dict()
    X1_t = dict()
    for k in np.arange(K):
        L_t[k] = np.zeros((p[k],p[k]))
        X0_t[k] = np.zeros((p[k],p[k]))
        X1_t[k] = np.zeros((p[k],p[k]))
     
     
    runtime = np.zeros(max_iter)
    kkt_residual = np.zeros(max_iter)
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
            
        eta_A = latent_ADMM_stopping_criterion(R_t, Theta_t, L_t, Lambda_t, X0_t, X1_t, S, G, lambda1, lambda2, mu1)
        kkt_residual[iter_t] = eta_A
            
        if eta_A <= eps_admm:
            status = 'optimal'
            break
        if verbose:
            print(f"------------Iteration {iter_t} of the ADMM Algorithm----------------")
        
        # R Update
        for k in np.arange(K):
            W_t = Theta_t[k] - L_t[k] - X0_t[k] - (1/rho) * S[k]
            eigD, eigQ = np.linalg.eigh(W_t)
            R_t[k] = phiplus(W_t, beta = 1/rho, D = eigD, Q = eigQ)
        
        # Theta Update
        for k in np.arange(K): 
            V_t = (R_t[k] + L_t[k] + X0_t[k] + Lambda_t[k] - X1_t[k]) * 0.5
            Theta_t[k] = prox_od_1norm(V_t, lambda1[k]/(2*rho))
        
        # L Update
        for k in np.arange(K):
            C_t = Theta_t[k] - X0_t[k] - R_t[k]
            eigD, eigQ = np.linalg.eigh(C_t)
            L_t[k] = prox_rank_norm(C_t, mu1[k]/rho, D = eigD, Q = eigQ)
            
        # Lambda Update
        for k in np.arange(K): 
            Z_t[k] = Theta_t[k] + X1_t[k]
            
        Lambda_t = prox_2norm_G(Z_t, G, lambda2/rho)
        
        # X Update
        for k in np.arange(K):
            X0_t[k] +=  R_t[k] - Theta_t[k] +L_t[k]
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
        assert abs(R_t[k].T - R_t[k]).max() <= 1e-5, "Solution is not symmetric"
        assert abs(Theta_t[k].T - Theta_t[k]).max() <= 1e-5, "Solution is not symmetric"
        assert abs(L_t[k].T - L_t[k]).max() <= 1e-5, "Solution is not symmetric"
    
    sol = {'R': R_t, 'Theta': Theta_t, 'L': L_t, 'X0': X0_t, 'X1': X1_t}
    if measure:
        info = {'status': status , 'runtime': runtime[:iter_t], 'kkt_residual': kkt_residual[1:iter_t +1]}
    else:
        info = {'status': status}
               
    return sol, info

def latent_ADMM_stopping_criterion(R, Theta, L, Lambda, X0, X1, S , G, lambda1, lambda2, mu1):
    
    K = len(S.keys())
    term1 = np.zeros(K)
    term2 = np.zeros(K)
    term3 = np.zeros(K)
    term4 = np.zeros(K)
    term5 = np.zeros(K)
    term6 = np.zeros(K)
    V = dict()
    
    for k in np.arange(K):
        eigD, eigQ = np.linalg.eigh(R[k] - S[k] - X0[k])
        proxk = phiplus(R[k] - S[k] - X0[k], beta = 1, D = eigD, Q = eigQ)
        term1[k] = np.linalg.norm(R[k] - proxk) / (1 + np.linalg.norm(R[k]))
        
        term2[k] = np.linalg.norm(Theta[k] - prox_od_1norm(Theta[k] + X0[k] - X1[k] , lambda1[k])) / (1 + np.linalg.norm(Theta[k]))
        
        eigD, eigQ = np.linalg.eigh(L[k] - X0[k])
        proxk = prox_rank_norm(L[k] - X0[k], beta = mu1[k], D = eigD, Q = eigQ)
        term6[k] = np.linalg.norm(L[k] - proxk) / (1 + np.linalg.norm(L[k]))
        
        term4[k] = np.linalg.norm(R[k] - Theta[k] + L[k]) / (1 + np.linalg.norm(Theta[k]))
        term5[k] = np.linalg.norm(Lambda[k] - Theta[k]) / (1 + np.linalg.norm(Theta[k]))
    
        V[k] = Lambda[k] + X1[k]
        
    V = prox_2norm_G(V, G, lambda2)
    for k in np.arange(K):
        term3[k] = np.linalg.norm(V[k] - Lambda[k]) / (1 + np.linalg.norm(Lambda[k]))
    
    res = max(np.linalg.norm(term1), np.linalg.norm(term2), np.linalg.norm(term3), np.linalg.norm(term4), np.linalg.norm(term5), np.linalg.norm(term6))
    return res




    