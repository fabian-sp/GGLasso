import numpy as np

from ggl_helper import moreau_h, moreau_P, construct_gamma, construct_jacobian_prox_p, Y_t, hessian_Y,  Phi_t
                        
from basic_linalg import t, Gdot, cg_general
#%%
def get_ppa_sub_params_default():
    ppa_sub_params = {'lambda1' : .1 , 'lambda2' : .1, 'sigma_t' : 1e8, 
          'eta' : .5, 'tau' : .5, 'rho' : .5, 'mu' : .25,
          'eps_t' : .5, 'delta_t' : .5} 
    
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

def PPA_subproblem(Omega_t, Theta_t, X_t, S, ppa_sub_params = None, verbose = False):
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
    
    while not(condA and condB):
        
        # step 0: set variables
        W_t = Omega_t - (sigma_t * (S + X_t))  
        V_t = Theta_t + (sigma_t * X_t)
        
        funY_Xt, gradY_Xt = Y_t( X_t, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t)
        
        eigD, eigQ = np.linalg.eig(W_t)
        print("Eigendecomposition is executed")
        Gamma = construct_gamma(W_t, sigma_t, D = eigD, Q = eigQ)
        
        W = construct_jacobian_prox_p( (1/sigma_t) * V_t, lambda1 , lambda2)
        
        # step 1: CG method
        kwargs = {'Gamma' : Gamma, 'eigQ': eigQ, 'W': W, 'sigma_t': sigma_t}
        cg_accur = min(eta, np.linalg.norm(gradY_Xt)**(1+tau))
        if verbose:
            print("Start CG method")
        D = cg_general(hessian_Y, Gdot, - gradY_Xt, eps = cg_accur, kwargs = kwargs)
        
        # step 2: line search 
        if verbose:
            print("Start Line search")
        alpha = rho
        Y_t_new = Y_t( X_t + alpha * D, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t)[0]
        while Y_t_new < funY_Xt + mu * alpha * Gdot(gradY_Xt , D):
            alpha *= rho
            Y_t_new = Y_t( X_t + alpha * D, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t)[0]
            
        # step 3: update variables and check stopping condition
        if verbose:
            print("Update primal-dual variables")
        X_t += alpha * D 
        
        X_sol = X_t.copy()
        
        Omega_sol = np.zeros((K,p,p))
        for k in np.arange(K):
            _, phip_k, _ = moreau_h( Omega_t[k,:,:] - sigma_t* (S[k,:,:] + X_sol[k,:,:]) , sigma_t)
            Omega_sol[k,:,:] = phip_k
        
        _, Theta_sol = moreau_P(Theta_t + sigma_t * X_sol, sigma_t * lambda1, sigma_t * lambda2)
        
        # step 4: evaluate stopping criterion
        opt_dist = Phi_t(Omega_sol, Theta_sol, S, Omega_t, Theta_t, sigma_t, lambda1, lambda2) - Y_t_new
        condA = opt_dist <= eps_t**2/(2*sigma_t)
        condB = opt_dist <= delta_t**2/(2*sigma_t) * ((np.linalg.norm(Omega_sol - Omega_t)**2 + np.linalg.norm(Theta_sol - Theta_t)**2))
    
    
    

    return Omega_sol, Theta_sol, X_sol


#%%
def PPDNA(S, lambda1, lambda2, Omega_0, Theta_0, sigma_0 = 10, max_iter = 100, verbose = False):
    """
    This is the outer proximal point algorithm
    Algorithm 2 in Zhang et al.
    """
    
    assert Omega_0.shape == Theta_0.shape == S.shape
    assert S.shape[1] == S.shape[2]
    
    (K,p,p) = S.shape
    
    # initialize 
    Omega_t = Omega_0.copy()
    Theta_t = Theta_0.copy()
    X_t0 = np.zeros((K,p,p))
    
    ppa_sub_params = get_ppa_sub_params_default()
    
    ppa_sub_params['sigma_t'] = sigma_0
    ppa_sub_params['lambda1'] = lambda1
    ppa_sub_params['lambda2'] = lambda2
    
    
    for iter_t in np.arange(max_iter):
        
        print(f"------------Iteration {iter_t} of the Proximal Point Algorithm----------------")
    
        Omega_t, Theta_t, X_t = PPA_subproblem(Omega_t, Theta_t, X_t0, S, ppa_sub_params = None, verbose = verbose)
        
        ppa_sub_params['sigma_t'] = 1.3 * ppa_sub_params['sigma_t']
        ppa_sub_params['eps_t'] = 0.5 * ppa_sub_params['eps_t']
        ppa_sub_params['delta_t'] = 0.5 * ppa_sub_params['delta_t']
        
        if verbose:
            print("sigma_t value: " , ppa_sub_params['sigma_t'])


    return Omega_t, Theta_t, X_t















