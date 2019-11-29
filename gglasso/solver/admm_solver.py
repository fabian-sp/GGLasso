"""
author: Fabian Schaipp
"""

import numpy as np

from ..helper.basic_linalg import trp,Gdot,Sdot
from .ggl_helper import prox_p, phiplus


def ADMM_MGL(S, lambda1, lambda2, Omega_0, reg, n_samples = None, rho = 1, max_iter = 100, eps_admm = 1e-5 , verbose = False):
    """
    This is an ADMM algorithm for solving the Multiple Graphical Lasso problem
    reg specifies the type of penalty, i.e. Group or Fused Graphical Lasso
    see also the article from Danaher et. al.
    
    n_samples are the sample sizes for the K instances, can also be None or integer to be 
    """
    assert Omega_0.shape == S.shape
    assert S.shape[1] == S.shape[2]
    assert reg in ['GGL', 'FGL']
        
    (K,p,p) = S.shape
    
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
    Omega_t = Omega_0.copy()
    Theta_t = np.zeros((K,p,p))
    X_t = np.zeros((K,p,p))
    
    for iter_t in np.arange(max_iter):
        
        eta_A = ADMM_stopping_criterion(Omega_t, Theta_t, X_t, S , lambda1, lambda2, nk, reg)
        if eta_A <= eps_admm:
            status = 'optimal'
            break
        
        print(f"------------Iteration {iter_t} of the ADMM Algorithm----------------")
            
        # Omega Xpdate
        W_t = Theta_t - X_t - (nk/rho) * S
        eigD, eigQ = np.linalg.eig(W_t)
        
        for k in np.arange(K):
            Omega_t[k,:,:] = phiplus(W_t[k,:,:], beta = nk[k,0,0]/rho, D = eigD[k,:], Q = eigQ[k,:,:])
        
        # Theta Xpdate
        Theta_t = prox_p(Omega_t + X_t, (1/rho)*lambda1, (1/rho)*lambda2, reg)
        # X Xpdate
        X_t = X_t + Omega_t - Theta_t
        
        if verbose:
            print(f"Current accuracy: ", eta_A)
        
    if eta_A > eps_admm:
        status = 'max iterations reached'
        
    print(f"ADMM terminated after {iter_t} iterations with accuracy {eta_A}")
    print(f"ADMM status: {status}")
               
    return Omega_t, status

def ADMM_stopping_criterion(Omega, Theta, X, S , lambda1, lambda2, nk, reg):
    
    assert Omega.shape == Theta.shape == S.shape
    assert S.shape[1] == S.shape[2]
    
    (K,p,p) = S.shape
    
    term1 = np.linalg.norm(Theta- prox_p(Theta + X , l1 = lambda1, l2= lambda2, reg = reg)) / (1 + np.linalg.norm(Theta))
    
    term2 = np.linalg.norm(Theta - Omega) / (1 + np.linalg.norm(Theta))
    
    proxK = np.zeros((K,p,p))
    eigD, eigQ = np.linalg.eig(Omega - nk*S - X)
    for k in np.arange(K):       
        proxK[k,:,:] = phiplus(A = Omega[k,:,:] - nk[k,0,0]*S[k,:,:] - X[k,:,:], beta = nk[k,0,0], D = eigD[k,:], Q = eigQ[k,:,:])
        
    term3 = np.linalg.norm(Omega - proxK) / (1 + np.linalg.norm(Omega))

    return max(term1, term2, term3)


    