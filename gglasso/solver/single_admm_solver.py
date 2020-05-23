"""
author: Fabian Schaipp
"""

import numpy as np
import time

from .ggl_helper import prox_od_1norm, phiplus, prox_rank_norm


def ADMM_SGL(S, lambda1, Omega_0 , Theta_0 = np.array([]), X_0 = np.array([]), \
             eps_admm = 1e-5 , verbose = False, measure = False, latent = False, mu1 = None, **kwargs):
    """
    This is an ADMM algorithm for solving the Single Graphical Lasso problem
    
    Omega_0 : start point -- must be specified as a (p,p) array
    S : empirical covariance matrix -- must be specified as a (p,p) array
    
    latent: boolean to indidate whether low rank term should be estimated
    mu1: low rank penalty paramater, if latent=True
    
    max_iter and rho can be specified via kwargs
    
    In the code, X are the SCALED dual variables, for the KKT stop criterion they have to be unscaled again!
    """
    assert Omega_0.shape == S.shape
    assert S.shape[0] == S.shape[1]
    assert lambda1 > 0
        
    if latent:
        assert mu1 is not None
        assert mu1 > 0
        
    (p,p) = S.shape
    
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
    if len(Theta_0) == 0:
        Theta_0 = Omega_0.copy()
    if len(X_0) == 0:
        X_0 = np.zeros((p,p))

    Theta_t = Theta_0.copy()
    L_t = np.zeros((p,p))
    X_t = X_0.copy()
       
    runtime = np.zeros(max_iter)
    kkt_residual = np.zeros(max_iter)
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
            
        eta_A = ADMM_stopping_criterion(Omega_t, Theta_t, L_t, rho*X_t, S , lambda1, latent, mu1)
        kkt_residual[iter_t] = eta_A
            
        if eta_A <= eps_admm:
            status = 'optimal'
            break
        if verbose:
            print(f"------------Iteration {iter_t} of the ADMM Algorithm----------------")
        
        # Omega Update
        W_t = Theta_t - L_t - X_t - (1/rho) * S
        eigD, eigQ = np.linalg.eigh(W_t)
        Omega_t= phiplus(W_t, beta = 1/rho, D = eigD, Q = eigQ)
        
        # Theta Update
        Theta_t = prox_od_1norm(Omega_t + L_t + X_t, (1/rho)*lambda1)
        
        # L Update
        if latent:
            C_t = Theta_t - X_t - Omega_t
            #C_t = (C_t.T + C_t)/2
            eigD, eigQ = np.linalg.eigh(C_t)
            L_t = prox_rank_norm(C_t, mu1/rho, D = eigD, Q = eigQ)

            
        # X Update
        X_t = X_t + Omega_t - Theta_t + L_t
        
        if measure:
            end = time.time()
            runtime[iter_t] = end-start
            
        if verbose:
            print(f"Current accuracy: ", eta_A)
        
    if eta_A > eps_admm:
        status = 'max iterations reached'
        
    print(f"ADMM terminated after {iter_t} iterations with accuracy {eta_A}")
    print(f"ADMM status: {status}")
    
    assert abs((Omega_t).T- Omega_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs((Theta_t).T- Theta_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs((L_t).T- L_t).max() <= 1e-5, "Solution is not symmetric"
    
    D,_ = np.linalg.eigh(Theta_t-L_t)
    if D.min() <= 0:
        print(f"WARNING: Theta (Theta - L resp.) is not positive definite. Solve to higher accuracy! (min EV is {D.min()})")
    
    D,_ = np.linalg.eigh(L_t)
    if D.min() < -1e-8:
        print(f"WARNING: L is not positive semidefinite. Solve to higher accuracy! (min EV is {D.min()})")
    
    if latent:
        sol = {'Omega': Omega_t, 'Theta': Theta_t, 'L': L_t, 'X': X_t}
    else:    
        sol = {'Omega': Omega_t, 'Theta': Theta_t, 'X': X_t}
        
    if measure:
        info = {'status': status , 'runtime': runtime[:iter_t], 'kkt_residual': kkt_residual[1:iter_t +1]}
    else:
        info = {'status': status}
               
    return sol, info

def ADMM_stopping_criterion(Omega, Theta, L, X, S , lambda1, latent = False, mu1 = None):
    
    assert Omega.shape == Theta.shape == S.shape
    assert S.shape[0] == S.shape[1]
    
    if not latent:
        assert np.all(L==0)
        
    (p,p) = S.shape
    
    term1 = np.linalg.norm(Theta - prox_od_1norm(Theta + X , l = lambda1)) / (1 + np.linalg.norm(Theta))
    
    term2 = np.linalg.norm(Omega - Theta + L) / (1 + np.linalg.norm(Theta))
    
    eigD, eigQ = np.linalg.eigh(Omega - S - X)
    proxO = phiplus(A = Omega - S - X, beta = 1, D = eigD, Q = eigQ)
    term3 = np.linalg.norm(Omega - proxO) / (1 + np.linalg.norm(Omega))
    
    term4 = 0
    if latent:
        eigD, eigQ = np.linalg.eigh(L - X)
        proxL = prox_rank_norm(A = L - X, beta = mu1, D = eigD, Q = eigQ)
        term4 = np.linalg.norm(L - proxL) / (1 + np.linalg.norm(L))
    
    return max(term1, term2, term3, term4)


    