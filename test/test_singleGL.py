"""
author: Fabian Schaipp

"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.covariance import GraphicalLasso
from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL


p = 10
N = 100

Sigma, Theta = group_power_network(p, K=5, M=2)

S, samples = sample_covariance_matrix(Sigma, N)


S = S[0,:,:]
Theta = Theta[0,:,:]



lambda1 = 0.01

singleGL = GraphicalLasso(alpha = lambda1, tol = 1e-6, max_iter = 500, verbose = True)
model = singleGL.fit(samples[0,:,:].T)

res_scikit = model.precision_


Omega_0 = np.eye(p)
sol, info = ADMM_SGL(S, lambda1, Omega_0 , eps_admm = 1e-4 , verbose = True, latent = False)


plt.scatter(np.triu(sol['Theta']), np.triu(res_scikit))

print( np.linalg.norm(sol['Theta'] - res_scikit) )

#%%
# test with low rank component
#from lvsglasso.lvsglasso import admm_consensus



Omega_0 = np.eye(p)
mu1 = .01

sol, info = ADMM_SGL(S, lambda1, Omega_0 , eps_admm = 1e-4 , verbose = True, latent = True, mu1 = mu1)


# rTheta, rL, _, _, _ , _ , _ = admm_consensus(S[np.newaxis,:,:], lambda1, mu1, niter=200, alpha=1., mu=None, 
#                    mu_cont=None, mu_cont_iter=10, mu_min=1e-6,
#                    S_init=None, L_init=None,
#                    abstol=1e-5, reltol=1e-5, verbose=True, compute_obj=False,
#                    compute_infeas=False, do_lowrank=True, do_sparse=True)



# plt.scatter(np.triu(sol['Theta']), np.triu(rTheta))


#%%
# teest for low rank component
# v = np.random.randn(p) 
# v = v/np.linalg.norm(v)
# L = np.outer(v,v)


# L = L * 0.5 * np.linalg.eigh(Theta)[0].min()

# R = Theta-L
# Sigma = np.linalg.pinv(R)
# assert np.linalg.eigh(R)[0].min() >= -1e-8

# S,_ = sample_covariance_matrix(Sigma[np.newaxis,:,:], N)
# S = S.squeeze()


# lambda1 = 0.08
# Omega_0 = np.eye(p)
# sol, info = ADMM_SGL(S, lambda1, Omega_0 , eps_admm = 1e-4 , verbose = True, latent = True, mu1 = .2)

# np.linalg.matrix_rank(sol['L'])

# fig,axs = plt.subplots(1,2)
# sns.heatmap(L, ax = axs[0], cmap = 'coolwarm')
# sns.heatmap(sol['L'], ax = axs[1], cmap = 'coolwarm')