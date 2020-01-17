"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""

import numpy as np

from gglasso.solver.ppdna_solver import PPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.latent_admm_solver import latent_ADMM_GGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.helper.experiment_helper import get_K_identity

p = 20
K = 5
N = 200
M = 2

reg = 'GGL'

if reg == 'GGL':
    Sigma, Theta = group_power_network(p, K, M)
elif reg == 'FGL':
    Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)

S, samples = sample_covariance_matrix(Sigma, N)

S = get_K_identity(K,p)

lambda1= 0.05
lambda2 = 0.05

Omega_0 = np.zeros((K,p,p))


solPPDNA, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, eps_ppdna = 1e-3 , verbose = True, measure = True)

solADMM, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, n_samples = None, eps_admm = 1e-4 , verbose = True)

solADMM, info = latent_ADMM_GGL(S, lambda1, lambda2, 1e-5, 1e-5, Omega_0, n_samples = None, eps_admm = 1e-5 , verbose = True, measure = False, max_iter = 100)



#%%

from gglasso.solver.ext_admm_solver import ext_ADMM_MGL, construct_G

p = 10
K = 5
lambda1 = .01
lambda2 = .01

S = dict()
Omega_0 = dict()

for k in np.arange(K):
    tmp = .1*np.random.rand(p,p)
    S[k] = np.eye(p) + (tmp.T@tmp)
    Omega_0[k] = np.zeros((p,p))
    
G = construct_G(p, K)

    
sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL' , Omega_0, G, eps_admm = 1e-5 , verbose = True, measure = False, max_iter = 30)

Theta_sol = np.zeros((K,p,p))
for k in np.arange(K):
    Theta_sol[k] = sol['Theta'][k]

