"""
 file for testing the solver
 
"""

import numpy as np
import scipy as sp

from basic_linalg import t
from ggl_solver import PPDNA

#%% inputs 

K = 5
p = 10
N = 100

tmp = np.random.normal(size=(p,p))
Sigma_inv = tmp.T @ tmp
Sigma_inv[Sigma_inv <= 2] = 0

L,_ = np.linalg.eig(Sigma_inv)

assert L.min() >= 0

Sigma = np.linalg.inv(Sigma_inv)


sample = np.zeros((K,p,N))
for k in np.arange(K):
    sample[k,:,:] = np.random.multivariate_normal(np.zeros(p), Sigma, N).T

S = np.zeros((K,p,p))
for k in np.arange(K):
    S[k,:,:] = np.cov(sample[k,:,:])
    
#S = np.tile(np.eye(p), (K,1,1))

#%%
Omega_0 = np.zeros((K,p,p))
Theta_0 =  np.zeros((K,p,p))

lambda1 = .01
lambda2 = .01

Omega_sol, Theta_sol, X_sol = PPDNA(S, lambda1, lambda2, Omega_0, Theta_0, sigma_0 = 10, max_iter = 10, verbose = True)
