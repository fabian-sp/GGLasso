"""
author: Fabian Schaipp
"""

import numpy as np
import time

from ..helper.basic_linalg import trp, Gdot
from .ggl_helper import prox_p, phiplus, prox_rank_norm, f, P_val


def ADMM_MGL(S, lambda1, lambda2, reg , Omega_0 , \
             Theta_0 = np.array([]), X_0 = np.array([]), n_samples = None, \
             eps_admm = 1e-5 , rho= 1., max_iter = 1000, verbose = False, measure = False, latent = False, mu1 = None):
    """
    This is an ADMM algorithm for solving the Multiple Graphical Lasso problem
    reg specifies the type of penalty, i.e. Group or Fused Graphical Lasso
    see also the article from Danaher et. al.
    
    Omega_0 : start point -- must be specified as a (K,p,p) array
    S : empirical covariance matrices -- must be specified as a (K,p,p) array
    
    n_samples are the sample sizes for the K instances, can also be None or integer
    
    latent: boolean to indidate whether low rank term should be estimated
    mu1: low rank penalty parameter (if latent=True), can be a vector of length K or a float
    
    
    In the code, X are the SCALED dual variables, for the KKT stop criterion they have to be unscaled again!
    """
    assert Omega_0.shape == S.shape
    assert S.shape[1] == S.shape[2]
    assert reg in ['GGL', 'FGL']
    assert min(lambda1, lambda2) > 0
        
    (K,p,p) = S.shape
    
    assert rho > 0, "ADMM penalization parameter must be positive."
    
    if latent:
        if type(mu1) == np.float64 or type(mu1) == float:
             mu1 = mu1*np.ones(K)
            
        assert mu1 is not None
        assert np.all(mu1 > 0)
    
    
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
    if len(Theta_0) == 0:
        Theta_0 = Omega_0.copy()
    if len(X_0) == 0:
        X_0 = np.zeros((K,p,p))

    Theta_t = Theta_0.copy()
    L_t = np.zeros((K,p,p))
    X_t = X_0.copy()
     
    runtime = np.zeros(max_iter)
    kkt_residual = np.zeros(max_iter)
    objective = np.zeros(max_iter)
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
            
        eta_A = ADMM_stopping_criterion(Omega_t, Theta_t, L_t, rho*X_t, S , lambda1, lambda2, nk, reg, latent = latent, mu1 = mu1)
        kkt_residual[iter_t] = eta_A
            
        if eta_A <= eps_admm:
            status = 'optimal'
            break
        if verbose:
            print(f"------------Iteration {iter_t} of the ADMM Algorithm----------------")
        
        # Omega Update
        W_t = Theta_t - L_t - X_t - (nk/rho) * S
        eigD, eigQ = np.linalg.eigh(W_t)
        
        for k in np.arange(K):
            Omega_t[k,:,:] = phiplus(W_t[k,:,:], beta = nk[k,0,0]/rho, D = eigD[k,:], Q = eigQ[k,:,:])
        
        # Theta Update
        Theta_t = prox_p(Omega_t + L_t + X_t, (1/rho)*lambda1, (1/rho)*lambda2, reg)
        
        #L Update
        if latent:
            C_t = Theta_t - X_t - Omega_t
            eigD, eigQ = np.linalg.eigh(C_t)
            for k in np.arange(K):
                L_t[k] = prox_rank_norm(C_t[k,:,:], mu1[k]/rho, D = eigD[k,:], Q = eigQ[k,:,:])
                
        # X Update
        X_t += Omega_t - Theta_t + L_t
        
        if measure:
            end = time.time()
            runtime[iter_t] = end-start
            objective[iter_t] = f(Omega_t, S) + P_val(Theta_t, lambda1, lambda2, reg)
            
        if verbose:
            print(f"Current accuracy: ", eta_A)
        
    if eta_A > eps_admm:
        status = 'max iterations reached'
        
    print(f"ADMM terminated after {iter_t} iterations with accuracy {eta_A}")
    print(f"ADMM status: {status}")
    
    assert abs(trp(Omega_t)- Omega_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs(trp(Theta_t)- Theta_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs(trp(L_t)- L_t).max() <= 1e-5, "Solution is not symmetric"
    
    D,_ = np.linalg.eigh(Theta_t - L_t)
    if D.min() <= 0:
        print("WARNING: Theta (Theta - L resp.) is not positive definite. Solve to higher accuracy!")
    
    D,_ = np.linalg.eigh(L_t)
    if D.min() < -1e-5:
        print("WARNING: L is not positive semidefinite. Solve to higher accuracy!")
    
    sol = {'Omega': Omega_t, 'Theta': Theta_t, 'L': L_t, 'X': X_t}
    if measure:
        info = {'status': status , 'runtime': runtime[:iter_t], 'kkt_residual': kkt_residual[1:iter_t +1], 'objective': objective[:iter_t]}
    else:
        info = {'status': status}
               
    return sol, info

def ADMM_stopping_criterion(Omega, Theta, L, X, S , lambda1, lambda2, nk, reg, latent = False, mu1 = None):
    
    assert Omega.shape == Theta.shape == S.shape
    assert S.shape[1] == S.shape[2]
    
    if not latent:
        assert np.all(L == 0)
    
    (K,p,p) = S.shape
    
    term1 = np.linalg.norm(Theta - prox_p(Theta + X , l1 = lambda1, l2= lambda2, reg = reg)) / (1 + np.linalg.norm(Theta))
    
    term2 = np.linalg.norm(Theta - Omega - L) / (1 + np.linalg.norm(Theta))
    
    proxK = np.zeros((K,p,p))
    eigD, eigQ = np.linalg.eigh(Omega - nk*S - X)
    for k in np.arange(K):       
        proxK[k,:,:] = phiplus(A = Omega[k,:,:] - nk[k,0,0]*S[k,:,:] - X[k,:,:], beta = nk[k,0,0], D = eigD[k,:], Q = eigQ[k,:,:])
     
    term3 = np.linalg.norm(Omega - proxK) / (1 + np.linalg.norm(Omega))
    
    if latent:
        proxL = np.zeros((K,p,p))
        eigD, eigQ = np.linalg.eigh(L - X)
        for k in np.arange(K):
            proxL[k,:,:] = prox_rank_norm(L[k,:,:] - X[k,:,:], beta = mu1[k], D = eigD[k,:], Q = eigQ[k,:,:])
        
        term4 = np.linalg.norm(L - proxL) / (1 + np.linalg.norm(L))
    else:
        term4 = 0
        
    return max(term1, term2, term3, term4)

    