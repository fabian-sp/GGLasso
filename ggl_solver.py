"""
author: Fabian Schaipp
"""

import numpy as np

from ggl_helper import t, moreau_h, moreau_P, construct_gamma, construct_jacobian_prox_p, eval_jacobian_phiplus,\
                        eval_jacobian_prox_p, Gdot



#%%
def Y_t( X, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t):
  
    assert min(lambda1, lambda2, sigma_t) > 0 , "at least one parameter is not positive"
    assert X.shape[1] == X.shape[2], "dimensions are not as expected"
  
    (K,p,p) = X.shape
  
    W_t = Omega_t - (sigma_t * (S + X))  
    V_t = Theta_t + (sigma_t * X)

    eigD, eigQ = np.linalg.eig(W_t)
  
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

#%%
    
#W_t = np.zeros((K,p,p))

eigD, eigQ = np.linalg.eig(W_t)
Gamma = construct_gamma(W_t, sigma_t, D = eigD, Q = eigQ)

W = construct_jacobian_prox_p( (1/sigma_t) * V_t, lambda1 , lambda2)



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
    


#%%
K = 5
p = 100

X = np.random.normal( size = (K,p,p))
X = t(X) @ X

Omega = np.random.normal( size = (K,p,p))
Omega_t = t(Omega) @ Omega

Theta = np.random.normal( size = (K,p,p))
Theta_t = t(Theta) @ Theta

S = np.random.normal( size = (K,p,p))
S = t(S) @ S

D = np.random.normal( size = (K,p,p))
D = t(D) @ D


sigma_t = 10
lambda1 = .1
lambda2 = .1

W_t = Omega_t - (sigma_t * (S + X))  
V_t = Theta_t + (sigma_t * X)



  
#Y_fun, Y_grad = Y_t(Omega_t, Theta_t, S, X, lambda1, lambda2, sigma_t)


#%%
# test the CG method

B = hessian_Y(X, Gamma,eigQ, W, sigma_t)

kwargs = {'Gamma' : Gamma, 'eigQ' : eigQ, 'W' : W, 'sigma_t' : sigma_t }

X_recovered = cg_general(hessian_Y, Gdot, B, eps = 1e-6, kwargs = kwargs)

np.linalg.norm(X-X_recovered) / np.linalg.norm(X)





