"""
author: Fabian Schaipp
"""

import numpy as np

from ggl_helper import t, moreau_h, moreau_P, Gdot


#%%
def Y_t(Omega_t, Theta_t, S, X, lambda1, lambda2, sigma_t):
  
  assert min(lambda1, lambda2, sigma_t) > 0 , "at least one parameter is not positive"
  assert S.shape[1] == S.shape[2], "dimensions are not as expected"
  
  (K,p,p) = S.shape
  
  W_t = Omega_t - (sigma_t * (S + X))  
  V_t = Theta_t + (sigma_t * X)
  
  D , Q = np.linalg.eig(W_t)
  
  grad1 = np.zeros((K,p,p))
  term1 = 0
  for k in np.arange(K):
    Psi_h, phip, _ = moreau_h(W_t[k,:,:] , sigma_t, D = D[k,:] , Q = Q[k,:,:] )
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
K = 5
p = 10

X = np.random.normal( size = (K,p,p))
X = t(X) @ X

Omega = np.random.normal( size = (K,p,p))
Omega_t = t(Omega) @ Omega

Theta = np.random.normal( size = (K,p,p))
Theta_t = t(Theta) @ Theta

S = np.random.normal( size = (K,p,p))
S = t(S) @ S


sigma_t = 10
lambda1 = .1
lambda2 = .1

  
Y_fun, Y_grad = Y_t(Omega_t, Theta_t, S, X, lambda1, lambda2, sigma_t)
