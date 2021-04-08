"""
author: Fabian Schaipp

Set the working directory to the file location if you want to save the plots.

This is a script for investigating starting points for the ADMM solver.
"""
from time import time
import numpy as np

from sklearn.covariance import GraphicalLasso

from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix

from regain.covariance import GraphicalLasso

p = 100
K = 1
N = 200
M = 5


Sigma, Theta = group_power_network(p, K, M)
S, sample = sample_covariance_matrix(Sigma, N)

S = S[0,:,:]
Sinv = np.linalg.pinv(S, hermitian = True)

N_l1 = 5
l1_range = np.logspace(0,-2,N_l1)

#%%

ITER = np.zeros((N_l1,3))
RT = np.zeros((N_l1,3))

TOL = 1e-5
RTOL = 1e-5

for j in range(N_l1):
    
    l1 = l1_range[j]
    
    # construct start points
    id_p = np.eye(p)
    Sinv_t = Sinv.copy()
    Sinv_t[np.abs(Sinv_t) < l1] = 0
    
    # solve with start point = identity
    sol1, info1 = ADMM_SGL(S, lambda1 = l1, Omega_0 = id_p, rho=1., max_iter=1000, tol=TOL, rtol=RTOL,\
                         verbose=False, measure=True, latent=False, mu1=None)
    
    ITER[j,0] = len(info1['runtime'])
    RT[j,0] = info1['runtime'].sum()
    
    # solve with start point = S^{-1}
    sol2, info2 = ADMM_SGL(S, lambda1 = l1, Omega_0 = Sinv, rho=1., max_iter=1000, tol=TOL, rtol=RTOL,\
                         verbose=False, measure=True, latent=False, mu1=None)
    
    ITER[j,1] = len(info2['runtime'])
    RT[j,1] = info2['runtime'].sum()
    
    # solve with start point = thresholded S^{-1}
    sol3, info3 = ADMM_SGL(S, lambda1 = l1, Omega_0 = Sinv_t, rho=1., max_iter=1000, tol=TOL, rtol=RTOL,\
                         verbose=False, measure=True, latent=False, mu1=None)
        
    ITER[j,2] = len(info3['runtime'])
    RT[j,2] = info3['runtime'].sum()
    
print(RT)
