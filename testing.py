"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""

import numpy as np
from matplotlib import pyplot as plt


from gglasso.solver.ggl_solver import PPDNA
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.evaluation import discovery_rate, error
from gglasso.helper.experiment_helper import lambda_parametrizer


p = 20
K = 5
N = 200
M = 2


Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)

S = sample_covariance_matrix(Sigma, N)

lambda1, lambda2 = lambda_parametrizer( np.sqrt(0.1)/(2**.25) , np.sqrt(0.1)/(2**.25))

Omega_0 = np.zeros((K,p,p))
Omega_0 = np.linalg.pinv(S)
reg = 'FGL'

sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = None, sigma_0 = 10, max_iter = 20, \
                                            eps_ppdna = 1e-4 , verbose = True, measure = True)

Theta_sol = sol['Theta']

