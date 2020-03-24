"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""

import numpy as np

from gglasso.solver.ppdna_solver import PPDNA, warmPPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.latent_admm_solver import latent_ADMM_GGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.helper.experiment_helper import get_K_identity
from gglasso.helper.model_selection import single_range_search

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
#S = get_K_identity(K,p)

lambda1= 0.05
lambda2 = 0.05
L = np.logspace(-1,-3,5)

#%%


est_uniform, est_indv, range_stats = single_range_search(S, L, N, method = 'eBIC', latent = True, mu = L[:-2])

Omega_0 = get_K_identity(K,p)


solPPDNA, info = warmPPDNA(S, lambda1, lambda2, reg, Omega_0, eps = 1e-4 , verbose = True, measure = True)

solADMM, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, n_samples = None, eps_admm = 1e-4 , verbose = True)

#solADMM, info = latent_ADMM_GGL(S, lambda1, lambda2, 1e-5, 1e-5, Omega_0, n_samples = None, eps_admm = 1e-5 , verbose = True, measure = False, max_iter = 100)

#%%
# tests for the extended ADMM version

from gglasso.solver.ext_admm_solver import ext_ADMM_MGL
from gglasso.helper.ext_admm_helper import construct_trivial_G

Sdict = dict()
Omega_0 = dict()

for k in np.arange(K):
    Sdict[k] = S[k,:,:].copy()
    Omega_0[k] = np.eye(p)

# constructs the "trivial" groups, i.e. all variables present in all instances  
G = construct_trivial_G(p, K)

aic, bic, sp, est_uniform, est_indv, ix_uniform, ix_indv, ix_mu = single_range_search(Sdict, L, N, method = 'eBIC', latent = True, mu = L[:-2])


solext, info = ext_ADMM_MGL(Sdict, lambda1, lambda2/np.sqrt(K), 'GGL' , Omega_0, G, eps_admm = 1e-4 , verbose = True, measure = False, max_iter = 50)

for k in np.arange(K):
    print(np.linalg.norm(solext['Theta'][k] - solADMM['Theta'][k,:,:]))
       

