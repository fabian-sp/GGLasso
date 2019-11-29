"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""

import numpy as np
from matplotlib import pyplot as plt


from gglasso.solver.ggl_solver import PPDNA
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.evaluation import discovery_rate, error, draw_group_graph
from gglasso.helper.experiment_helper import lambda_parametrizer


p = 100
K = 5
N = 2000
M = 10


Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)

draw_group_graph(Theta)

S = sample_covariance_matrix(Sigma, N)
W1 = np.array([ 0.05, 0.1, 0.2 , 0.3])
ERR = np.zeros(len(W1))
FPR = np.zeros(len(W1))
TPR = np.zeros(len(W1))

for j in np.arange(len(W1)):
    lambda1, lambda2 = lambda_parametrizer(w1 = W1[j], w2 = 0.2)

    Omega_0 = np.zeros((K,p,p))
    Omega_0 = np.linalg.pinv(S)
    reg = 'FGL'
    
    sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = None, sigma_0 = 10, max_iter = 20, \
                                                eps_ppdna = 1e-4 , verbose = False)
    
    Theta_sol = sol['Theta']
    
    TPR[j] = discovery_rate(Theta_sol, Theta, t = 1e-5)['TPR']
    FPR[j] = discovery_rate(Theta_sol, Theta, t = 1e-5)['FPR']
    ERR[j] = error(Theta_sol, Theta)


plt.figure()
plt.plot(FPR, TPR)
