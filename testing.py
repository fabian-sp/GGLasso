"""
 file for testing the solver
 
"""

import numpy as np
import scipy as sp

from basic_linalg import t
from ggl_solver import PPDNA

#%% inputs 

K = 5
p = 100
N = 200

tmp = np.random.normal(size=(p,p))
Sigma_inv = tmp.T @ tmp
Sigma_inv[Sigma_inv <= 2] = 0

#np.fill_diagonal(Sigma_inv , 1)
Sigma = np.linalg.inv(Sigma_inv)

L,_ = np.linalg.eig(Sigma)

assert L.min() >= 0


sample = np.zeros((K,p,N))
for k in np.arange(K):
    sample[k,:,:] = np.random.multivariate_normal(np.zeros(p), Sigma, N).T

S = np.zeros((K,p,p))
for k in np.arange(K):
    S[k,:,:] = np.cov(sample[k,:,:])
    
#S = np.tile(np.eye(p), (K,1,1))


diag_S = 1/np.diagonal(S, axis1 = 1, axis2 = 2)
Omega_0 = np.apply_along_axis(np.diag, 1,diag_S)
#Omega_0 = np.zeros((K,p,p))

Theta_0 = Omega_0.copy()
#%%

lambda1 = .5e-2
lambda2 = .5e-2

Omega_sol, Theta_sol, X_sol = PPDNA(S, lambda1, lambda2, Omega_0, Theta_0, sigma_0 = 10, max_iter = 100, eps_ppdna = 1e-5, verbose = True)


naive = np.linalg.inv(S[0,:,:])