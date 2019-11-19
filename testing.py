"""
 file for testing the solver
 
"""

import numpy as np
import scipy as sp
from time import time

from gglasso.basic_linalg import trp
from gglasso.ggl_solver import PPDNA

from evaluation import discovery_rate, draw_group_graph

#%% inputs 

K = 5
p = 20
N = 20

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

lambda1 = 1e-2
lambda2 = 1e-2

start = time()
Omega_sol, Theta_sol, X_sol = PPDNA(S, lambda1, lambda2, Omega_0, Theta_0, reg = 'GGL', sigma_0 = 10, max_iter = 100, eps_ppdna = 1e-5, verbose = True)
end = time()
print("Running time was ", end-start, "sec")

naive  = np.linalg.inv(S)

# Sanity check 1: for lambda very small, we want to recover S^-1
print(np.linalg.norm(naive-Theta_sol)/np.linalg.norm(naive))

#%%

S_true = np.tile(Sigma_inv, (K,1,1))

fig = draw_group_graph(Theta_sol , t = 1e-9)

fig = draw_group_graph(S_true , t = 1e-9)

discovery_rate(Theta_sol , S_true, t = 1e-9)
