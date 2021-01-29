"""
author: Fabian Schaipp
"""

import numpy as np
import time

from ..helper.basic_linalg import trp
from .ggl_helper import prox_p, phiplus, prox_chi, prox_rank_norm


def latent_ADMM_GGL(S, lambda1, lambda2, mu1, mu2, R_0, \
             n_samples = None, eps_admm = 1e-5 , verbose = False, measure = False, **kwargs):
    """
    This is an ADMM algorithm for solving the Group Graphical Lasso problem with Latent Variables
    
    n_samples are the sample sizes for the K instances, can also be None or integer (currently not used!)
    max_iter and rho can be specified via kwargs
    In the code, X0,X1 are the SCALED dual variables, for the KKT stop criterion they have to be unscaled again!
    W denotes the additional variables from reformulation
    """
    assert R_0.shape == S.shape
    assert S.shape[1] == S.shape[2]
    assert min(lambda1, lambda2) > 0
    assert mu1 > 0
    assert mu2 >= 0
    
    (K,p,p) = S.shape
    
    if 'max_iter' in kwargs.keys():
        max_iter = kwargs.get('max_iter')
    else:
        max_iter = 1000
    if 'rho' in kwargs.keys():
        assert kwargs.get('rho') > 0
        rho = kwargs.get('rho')
    else:
        rho = 1.

    # n_samples None --> set them all to 1
    # n_samples integer --> all instances have same number of samples
    # else --> n_samples should be array with sample sizes
    if n_samples == None:
        nk = np.ones((K,1,1))
    elif type(n_samples) == int:
        nk = n_samples * np.ones((K,1,1)) 
    else:
        assert len(nk) == K
        nk = n_samples.reshape(K,1,1)
    
    # initialize 
    status = 'not optimal'
    R_t = R_0.copy()
        
    Theta_t = np.zeros((K,p,p))
    L_t = np.zeros((K,p,p))
    X0_t = np.zeros((K,p,p))
    X1_t = np.zeros((K,p,p))
    W_t = np.zeros((K,p,p))
    
    runtime = np.zeros(max_iter)
    kkt_residual = np.zeros(max_iter)
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
            
        eta_A = latent_ADMM_stopping_criterion(R_t, Theta_t, L_t, W_t, rho*X0_t, rho*X1_t, S , lambda1, lambda2, mu1, mu2)
        kkt_residual[iter_t] = eta_A
            
        if eta_A <= eps_admm:
            status = 'optimal'
            break
        if verbose:
            print(f"------------Iteration {iter_t} of the ADMM Algorithm with Latent Variables----------------")
            
        # R step
        A_t = Theta_t - L_t - X0_t
        H1_t = (A_t + trp(A_t))/2 - (1/rho) * S
        eigD, eigQ = np.linalg.eigh(H1_t) 
        for k in np.arange(K):
            R_t[k,:,:] = phiplus(H1_t[k,:,:], 1/rho, D = eigD[k,:], Q = eigQ[k,:,:])
            
        # Theta step
        B_t = (R_t + L_t + X0_t) 
        #Theta_t = prox_od_1norm(B_t, lambda1/rho)
        Theta_t = prox_p(B_t, lambda1/rho, lambda2/rho, reg = 'GGL')
        
        
        # L step
        C_t = (Theta_t - X0_t - R_t + W_t + X1_t) / 2
        H2_t = (C_t + trp(C_t))/2
        eigD, eigQ = np.linalg.eigh(H2_t)
        for k in np.arange(K):
            L_t[k,:,:] = prox_rank_norm(H2_t[k,:,:], mu1/(2*rho), D = eigD[k,:], Q = eigQ[k,:,:])
        
        
        # W step
        W_t = prox_chi(L_t - X1_t, mu2/rho)
        
        # dual variables
        X0_t += R_t - Theta_t + L_t
        X1_t += W_t - L_t
        
        if measure:
            end = time.time()
            runtime[iter_t] = end-start
            
        if verbose:
            print(f"Current accuracy: ", eta_A)
            
    if eta_A > eps_admm:
        status = 'max iterations reached'
        
    print(f"ADMM terminated after {iter_t} iterations with accuracy {eta_A}")
    print(f"ADMM status: {status}")
    
    assert abs(trp(R_t)- R_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs(trp(Theta_t)- Theta_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs(trp(L_t)- L_t).max() <= 1e-5, "Solution is not symmetric"
    
    sol = {'R': R_t, 'Theta': Theta_t, 'L': L_t, 'X0': X0_t, 'X1': X1_t}
    if measure:
        info = {'status': status , 'runtime': runtime[:iter_t], 'kkt_residual': kkt_residual[1:iter_t +1]}
    else:
        info = {'status': status}
        
    return sol, info
                
         
def latent_ADMM_stopping_criterion(R, Theta, L, W, X0, X1, S , lambda1, lambda2, mu1, mu2):
    
    assert R.shape == Theta.shape == S.shape
    assert S.shape[1] == S.shape[2]
    
    (K,p,p) = S.shape
    
    term1 = np.linalg.norm(Theta - prox_p(Theta + X0 , l1 = lambda1, l2= lambda2, reg = 'GGL')) / (1 + np.linalg.norm(Theta))
     
    proxK = np.zeros((K,p,p))
    eigD, eigQ = np.linalg.eigh(R - S - X0)
    for k in np.arange(K):       
        proxK[k,:,:] = phiplus(A = R[k,:,:] - S[k,:,:] - X0[k,:,:], beta = 1, D = eigD[k,:], Q = eigQ[k,:,:])
        
    term2 = np.linalg.norm(R - proxK) / (1 + np.linalg.norm(R))
    
    proxK = np.zeros((K,p,p))
    eigD, eigQ = np.linalg.eigh(L - X0 - X1)
    for k in np.arange(K):       
        proxK[k,:,:] = prox_rank_norm(A = L[k,:,:] - X0[k,:,:] - X1[k,:,:], beta = mu1, D = eigD[k,:], Q = eigQ[k,:,:])
    
    term3 = np.linalg.norm(L - proxK) / (1 + np.linalg.norm(L))
    
    term4 = np.linalg.norm(W - prox_chi(W + X1, mu2)) / (1 + np.linalg.norm(W))
    
    term5 = np.linalg.norm(R-Theta+L) / (1 + np.linalg.norm(R))
    term6 = np.linalg.norm(W-L) / (1 + np.linalg.norm(W))
    
    return max(term1, term2, term3, term4, term5, term6)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        