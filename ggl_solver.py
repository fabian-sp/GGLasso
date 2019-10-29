import numpy as np

from ggl_helper import moreau_h, moreau_P, construct_gamma, construct_jacobian_prox_p, eval_jacobian_phiplus,\
                        eval_jacobian_prox_p, Phi_t
                        
from basic_linalg import t, Gdot, cg_general
                        
#%%
def hessian_Y(D , Gamma, eigQ, W, sigma_t):
    """
    this is the linear operator for the CG method
    argument is D
    Gamma and W are constructed beforehand in order to evaluate more efficiently
    """
    tmp1 = eval_jacobian_phiplus( D, Gamma, eigQ)
    tmp2 = eval_jacobian_prox_p( D , W)

    res = - sigma_t * (tmp1 + tmp2)
    return res


def Y_t( X, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t):
  
    assert min(lambda1, lambda2, sigma_t) > 0 , "at least one parameter is not positive"
    assert X.shape[1] == X.shape[2], "dimensions are not as expected"
  
    (K,p,p) = X.shape
  
    W_t = Omega_t - (sigma_t * (S + X))  
    V_t = Theta_t + (sigma_t * X)

    eigD, eigQ = np.linalg.eig(W_t)
    print("Eigendecomposition is executed")
  
    grad1 = np.zeros((K,p,p))
    term1 = 0
    for k in np.arange(K):
        Psi_h, phip, _ = moreau_h(W_t[k,:,:] , sigma_t, D = eigD[k,:] , Q = eigQ[k,:,:] )
        term1 += (1/sigma_t) * Psi_h
        grad1[k,:,:] = phip
    
    term2 = - 1/(2*sigma_t) * ( Gdot(W_t, W_t) + Gdot(V_t, V_t))
    term3 = 1/(2*sigma_t) * (  Gdot(Omega_t, Omega_t)  +  Gdot(Theta_t, Theta_t)   )  
  
    Psi_P , U = moreau_P(V_t, sigma_t * lambda1, sigma_t*lambda2)  
    term4 = (1/sigma_t) * Psi_P
  
    fun = term1 + term2 + term3 + term4
    grad = grad1 - U
  
    return fun, grad

#%% inputs

K = 5
p = 10

sigma_t = 10
lambda1 = .1
lambda2 = .1

eta = 0.5
tau = 0.5
rho = 0.5
mu = 0.25


S = np.tile(np.eye(p), (K,1,1))
Omega_t = 2*S
Theta_t = 2*S

X_t =  np.tile(np.eye(p), (K,1,1))

max_iter = 10
eps_t = 0.5
delta_t = 0.5

condA = False
condB = False

#%% algorithm
#for j in np.arange(max_iter):
while not(condA & condB):
    
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
    D = cg_general(hessian_Y, Gdot, - gradY_Xt, eps = cg_accur, kwargs = kwargs)
    
    # step 2: line search 
    alpha = rho
    Y_t_new = Y_t( X_t + alpha * D, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t)[0]
    while Y_t_new < funY_Xt + mu * alpha * Gdot(gradY_Xt , D):
        alpha *= rho
        Y_t_new = Y_t( X_t + alpha * D, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t)[0]
        
    # step 3: update variables and check stopping condition
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
    
    
    




