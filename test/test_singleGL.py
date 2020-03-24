"""
author: Fabian Schaipp

"""

import numpy as np
from sklearn.covariance import GraphicalLasso
from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL


p = 10
N = 100

Sigma, Theta = group_power_network(p, K=1, M=1)

S, samples = sample_covariance_matrix(Sigma, N)


S = S.squeeze()
Theta = Theta.squeeze()


lambda1 = 0.01

singleGL = GraphicalLasso(alpha = lambda1, tol = 1e-6, max_iter = 500, verbose = True)
model = singleGL.fit(samples[0,:,:].T)

res_scikit = model.precision_


Omega_0 = np.eye(p)
sol, info = ADMM_SGL(S, lambda1, Omega_0 , eps_admm = 1e-4 , verbose = True, latent = True, mu1 = .1)


assert np.linalg.norm(sol['Theta'] - res_scikit) <= 1e-3
