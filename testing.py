"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""

import numpy as np
from matplotlib import pyplot as plt


from gglasso.solver.ggl_solver import PPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_parametrizer, discovery_rate, error


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

S = sample_covariance_matrix(Sigma, N)

lambda1= 0.1
lambda2 = 0.1

Omega_0 = np.zeros((K,p,p))


solPPDNA, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, sigma_0 = 10, max_iter = 20, \
                                            eps_ppdna = 1e-4 , verbose = True, measure = True)

solADMM, info = ADMM_MGL(S, lambda1, lambda2, Omega_0, reg, n_samples = None, rho = 1, max_iter = 100, eps_admm = 1e-4 , verbose = True)

Theta_sol = solPPDNA['Theta']

