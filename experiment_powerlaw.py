"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""

import numpy as np
from matplotlib import pyplot as plt


from gglasso.solver.ggl_solver import PPDNA
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.evaluation import discovery_rate, error

def lambda_parametrizer(w1 = 0.1, w2 = 0.2):
    
    l2 = np.sqrt(2) * w1 * w2
    l1 = w1 - (1/np.sqrt(2)) * l2
   
    return l1,l2
    
    

p = 20
K = 5
N = 200
M = 2


Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)


S = sample_covariance_matrix(Sigma, N)

lambda1, lambda2 = lambda_parametrizer(w1 = 0.1, w2 = 0.2)

Omega_0 = np.zeros((K,p,p))
Omega_0 = np.linalg.pinv(S)
reg = 'FGL'

Omega_sol, Theta_sol, X_sol, status = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = None, sigma_0 = 10, max_iter = 100, \
                                            eps_ppdna = 1e-4 , verbose = True)


discovery_rate(Theta_sol, Theta, t = 1e-5)
error(Theta_sol, Theta)
