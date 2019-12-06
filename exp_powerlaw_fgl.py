"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""
from time import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from inverse_covariance import QuicGraphicalLasso

from gglasso.solver.ggl_solver import PPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, aic, error, draw_group_graph, draw_group_heatmap


p = 100
K = 10
N = 2000
M = 10

reg = 'FGL'

Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)/np.linalg.norm(np.eye(p))

draw_group_heatmap(Theta)

S = sample_covariance_matrix(Sigma, N)
Sinv = np.linalg.pinv(S, hermitian = True)



lambda1 = 0.1
lambda2 = 0.1

solvers = ['PPDNA', 'ADMM', 'QUIC']

Omega_0 = np.zeros((K,p,p))
Theta_0 = np.zeros((K,p,p))
X_0 = np.zeros((K,p,p))

start = time()
sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, sigma_0 = 10, max_iter = 20, \
                                                        eps_ppdna = 1e-3, verbose = True)
end = time()

Theta_sol = sol['Theta']
Omega_sol = sol['Omega']


fig,axs = plt.subplots(nrows = 1, ncols = 2)
draw_group_heatmap(Theta, axs[0])
draw_group_heatmap(Theta_sol, axs[1])
