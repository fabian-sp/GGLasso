"""
author: Fabian Schaipp

This file contains the proximal point algorithm proposed by Zhang et al.
"""

import numpy as np
import time

from .ggl_helper import prox_p, phiplus, moreau_h, moreau_P, construct_gamma, construct_jacobian_prox_p, Y_t, hessian_Y,  Phi_t, f, P_val
from .admm_solver import ADMM_MGL
from ..helper.basic_linalg import trp,Gdot, cg_general

def get_ppdna_params(ppdna_params = None):
    
    # initialize if empty
    if ppdna_params == None:
        ppdna_params = {}
    p0 = list(ppdna_params.keys())
    
    if 'max_iter' not in p0:
        ppdna_params['max_iter'] = 20
    else:
        assert ppdna_params['max_iter'] > 0
    
    if 'sigma_0' not in p0:
        ppdna_params['sigma_0'] = 100
    else:
        assert ppdna_params['sigma_0'] > 0
        
    if 'sigma_fix' not in p0:
        ppdna_params['sigma_fix'] = False

    return ppdna_params
         

def get_ppa_sub_params_default():
    ppa_sub_params = { 'sigma_t' : 1e3, 
          'eta' : .5, 'tau' : .5, 'rho' : .5, 'mu' : .25,
          'eps_t' : .9, 'delta_t' : .9} 
    
    return ppa_sub_params

def check_ppa_sub_params(ppa_sub_params):
    
    assert ppa_sub_params['lambda1'] > 0
    assert ppa_sub_params['lambda2'] > 0
    assert ppa_sub_params['sigma_t'] > 0
    
    assert ppa_sub_params['mu'] > 0 and ppa_sub_params['mu'] < .5
    assert ppa_sub_params['eta'] > 0 and ppa_sub_params['eta'] < 1
    assert ppa_sub_params['tau'] > 0 and ppa_sub_params['tau'] <= 1
    assert ppa_sub_params['rho'] > 0 and ppa_sub_params['rho'] < 1
    
    assert ppa_sub_params['eps_t'] >= 0
    assert ppa_sub_params['delta_t'] >= 0 and ppa_sub_params['delta_t'] < 1
    
    return

def PPA_subproblem(Omega_t, Theta_t, X_t, S, reg, ppa_sub_params = None, verbose = False):
    """
    This is the dual based semismooth Newton method solver for the PPA subproblems
    Algorithm 1 in Zhang et al.
    """
    
    assert Omega_t.shape == Theta_t.shape == S.shape == X_t.shape
    assert S.shape[1] == S.shape[2]
    
    (K,p,p) = S.shape
    
    if ppa_sub_params == None:
        ppa_sub_params = get_ppa_sub_params_default()
        
    check_ppa_sub_params(ppa_sub_params)
    
    sigma_t = ppa_sub_params['sigma_t']
    lambda1 = ppa_sub_params['lambda1']
    lambda2 = ppa_sub_params['lambda2']
    
    eta = ppa_sub_params['eta']
    tau = ppa_sub_params['tau']
    rho = ppa_sub_params['rho']
    mu = ppa_sub_params['mu']
    eps_t = ppa_sub_params['eps_t']
    delta_t = ppa_sub_params['delta_t']

    condA = False
    condB = False
    
    sub_iter = 0
    
    while not(condA and condB) and sub_iter < 10:
        # step 0: set variables
        W_t = Omega_t - (sigma_t * (S + X_t))  
        V_t = Theta_t + (sigma_t * X_t)
        
        funY_Xt, gradY_Xt = Y_t( X_t, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t, reg)
        
        
        eigD, eigQ = np.linalg.eigh(W_t)
        if verbose:
            print("Eigendecomposition is executed in PPA_subproblem")
            
        Gamma = construct_gamma(W_t, sigma_t, D = eigD, Q = eigQ)
        W = construct_jacobian_prox_p( (1/sigma_t) * V_t, lambda1 , lambda2, reg)
        
        # step 1: CG method
        kwargs = {'Gamma' : Gamma, 'eigQ': eigQ, 'W': W, 'sigma_t': sigma_t}
        cg_accur = min(eta, np.linalg.norm(gradY_Xt)**(1+tau), 1e-10)
        if verbose:
            print("Start CG method")
        D = cg_general(hessian_Y, Gdot, - gradY_Xt, eps = cg_accur, kwargs = kwargs, verbose = verbose)
        
        # step 2: line search 
        if verbose:
            print("Start Line search")
        alpha = rho
        Y_t_new = Y_t( X_t + alpha * D, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t, reg)[0]
        while Y_t_new < funY_Xt + mu * alpha * Gdot(gradY_Xt , D):
            alpha *= rho
            Y_t_new = Y_t( X_t + alpha * D, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t, reg)[0]
            
        # step 3: update variables and check stopping condition
        if verbose:
            print("Update primal-dual variables")
        X_t += alpha * D 
        
        X_sol = X_t.copy()
        
        Omega_sol = np.zeros((K,p,p))
        eigW, eigV = np.linalg.eigh(Omega_t - sigma_t * (S + X_sol))
        for k in np.arange(K):
            _, phip_k, _ = moreau_h( Omega_t[k,:,:] - sigma_t * (S[k,:,:] + X_sol[k,:,:]) , sigma_t , eigW[k,:], eigV[k,:,:])
            Omega_sol[k,:,:] = phip_k
        
        _, Theta_sol = moreau_P(Theta_t + sigma_t * X_sol, sigma_t * lambda1, sigma_t * lambda2, reg)
        
        # step 4: evaluate stopping criterion
        opt_dist = Phi_t(Omega_sol, Theta_sol, S, Omega_t, Theta_t, sigma_t, lambda1, lambda2, reg) - Y_t_new
        condA = opt_dist <= eps_t**2/(2*sigma_t)
        condB = opt_dist <= delta_t**2/(2*sigma_t) * ((np.linalg.norm(Omega_sol - Omega_t)**2 + np.linalg.norm(Theta_sol - Theta_t)**2))
    
        sub_iter += 1
        
    if verbose and not(condA and condB):
        print("Subproblem could not be solve with the given accuracy! -- reached maximal iterations")
            
    
    return Omega_sol, Theta_sol, X_sol


def PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = np.array([]), X_0 = np.array([]), ppdna_params = None, eps_ppdna = 1e-5 , verbose = False, measure = False):
    """
    This is the outer proximal point algorithm
    Algorithm 2 in Zhang et al.
    """
    
    assert Omega_0.shape == S.shape
    assert S.shape[1] == S.shape[2]
    assert reg in ['GGL', 'FGL']
    assert min(lambda1, lambda2) > 0
    (K,p,p) = S.shape
    
    # initialize 
    status = 'not optimal'
    
    Omega_t = Omega_0.copy()
    if len(Theta_0) == 0:
        Theta_0 = Omega_0.copy()
    if len(X_0) == 0:
        X_0 = np.zeros((K,p,p))

    Theta_t = Theta_0.copy()
    X_t = X_0.copy()
    
    # adds all necessary paramters which are not given as input
    ppdna_params = get_ppdna_params(ppdna_params)
    max_iter = ppdna_params['max_iter']
    sigma_0 = ppdna_params['sigma_0']
    
    ppa_sub_params = get_ppa_sub_params_default()
    ppa_sub_params['sigma_t'] = sigma_0
    ppa_sub_params['lambda1'] = lambda1
    ppa_sub_params['lambda2'] = lambda2
    
    runtime = np.zeros(max_iter)
    kkt_residual = np.zeros(max_iter)
    objective = np.zeros(max_iter)
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
            
        # check stopping criterion
        eta_P = PPDNA_stopping_criterion(Omega_t, Theta_t, X_t, S , ppa_sub_params, reg)
        kkt_residual[iter_t] = eta_P
        
        if eta_P <= eps_ppdna:
            status = 'optimal'
            break
        
        if verbose:
            print(f"------------Iteration {iter_t} of the Proximal Point Algorithm----------------")
        
        Omega_t, Theta_t, X_t = PPA_subproblem(Omega_t, Theta_t, X_t, S, reg = reg, ppa_sub_params = ppa_sub_params, verbose = verbose)
        
        if measure:
            end = time.time()
            runtime[iter_t] = end-start
            objective[iter_t] = f(Omega_t, S) + P_val(Omega_t, lambda1, lambda2, reg)
            
        
        if not ppdna_params['sigma_fix']:
            ppa_sub_params['sigma_t'] = 1.3 * ppa_sub_params['sigma_t']
        ppa_sub_params['eps_t'] = 0.9 * ppa_sub_params['eps_t']
        ppa_sub_params['delta_t'] = 0.9 * ppa_sub_params['delta_t']
            
        if verbose:
            print("sigma_t value: " , ppa_sub_params['sigma_t'])
            print(f"Current accuracy: ", eta_P)
     
    if eta_P > eps_ppdna:
            status = 'max iterations reached'    
        
    print(f"PPDNA terminated after {iter_t} iterations with accuracy {eta_P}")
    print(f"PPDNA status: {status}")
    
    assert abs(trp(Omega_t)- Omega_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs(trp(Theta_t)- Theta_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs(trp(X_t)- X_t).max() <= 1e-5, "Solution is not symmetric"
    D,_ = np.linalg.eigh(Theta_t)
    if D.min() <= 0:
        print("WARNING: Theta is not positive definite. Solve to higher accuracy!")
    
    sol = {'Omega': Omega_t, 'Theta': Theta_t, 'X': X_t}
    if measure:
        # last runtime irrelevant (as break) and first residual irrelevant
        info = {'status': status , 'runtime': runtime[:iter_t], 'kkt_residual': kkt_residual[1:iter_t + 1], 'objective': objective[:iter_t]}
    else:
        info = {'status': status}
    return sol, info


def PPDNA_stopping_criterion(Omega, Theta, X, S , ppa_sub_params, reg): 
    assert Omega.shape == Theta.shape == S.shape
    assert S.shape[1] == S.shape[2]
    
    (K,p,p) = S.shape
    
    term1 = np.linalg.norm(Theta- prox_p(Theta + X , l1 = ppa_sub_params['lambda1'], l2= ppa_sub_params['lambda2'], reg = reg)) / (1 + np.linalg.norm(Theta))
    
    term2 = np.linalg.norm(Theta - Omega) / (1 + np.linalg.norm(Theta))
    
    proxK = np.zeros((K,p,p))
    eigD, eigQ = np.linalg.eigh(Omega-S-X)
    for k in np.arange(K):       
        proxK[k,:,:] = phiplus(A = Omega[k,:,:] - S[k,:,:] - X[k,:,:], beta = 1, D = eigD[k,:], Q = eigQ[k,:,:])
    
    term3 = np.linalg.norm(Omega - proxK) / (1 + np.linalg.norm(Omega))
    
    return max(term1, term2, term3)


def warmPPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = np.array([]), X_0 = np.array([]), admm_params = None, ppdna_params = None, eps = 1e-5 , eps_admm = 1e-3, verbose = False, measure = False):
    """
    function for solving a MGL problem with PPDNA but using ADMM as a warm start (i.e. solve to low accuracy with ADMM first)
    """
    
    if eps >= 1e-3:
        eps_admm = eps
        phase2 = False
    else:
        eps_ppdna = eps
        phase2 = True
    
    rho = 1.
    sol1, info1 = ADMM_MGL(S, lambda1, lambda2, reg , Omega_0 , Theta_0, X_0, n_samples = None, eps_admm = eps_admm , verbose = verbose, measure = measure, rho = rho)
    
    assert info1['status'] == 'optimal'
    
    Theta_0 = sol1['Theta']
    Omega_0 = sol1['Omega']
    # ADMM returns scaled dual variables --> rescale
    X_0 = rho*sol1['X']
    
    if phase2:
        sol2, info2 = PPDNA(S, lambda1, lambda2, reg, Omega_0 = Omega_0, Theta_0 = Theta_0, X_0 = X_0, ppdna_params = ppdna_params,  eps_ppdna = eps_ppdna , verbose = verbose, measure = measure)
    
        # append the infos
        if measure:
            info2['runtime'] = np.append(info1['runtime'], info2['runtime'])
            info2['kkt_residual'] = np.append(info1['kkt_residual'], info2['kkt_residual'])
            info2['objective'] = np.append(info1['objective'], info2['objective'])
            info2['iter_admm'] = len(info1['runtime']) -1 
            
        return sol2, info2
    
    else:
        return sol1, info1
    


