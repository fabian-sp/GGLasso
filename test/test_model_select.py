"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""

import numpy as np
import seaborn as sns

from gglasso.solver.ppdna_solver import PPDNA, warmPPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.latent_admm_solver import latent_ADMM_GGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.helper.experiment_helper import get_K_identity
from gglasso.helper.model_selection import grid_search, single_range_search

p = 20
K = 5
N = 2000
M = 2

reg = 'GGL'

if reg == 'GGL':
    Sigma, Theta = group_power_network(p, K, M)
elif reg == 'FGL':
    Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)

S, samples = sample_covariance_matrix(Sigma, N)
#S = get_K_identity(K,p)


L1 = np.logspace(0,-2, 8)
W2 = np.linspace(0.002, 0.001, 5)

#%%
latent = True
ix_mu = None

if latent:
    mu = np.linspace(.5,.05,5)
else:
    mu = None

est_uniform, est_indv, range_stats = single_range_search(S, L1, N, method = 'eBIC', latent = latent, mu = mu)

ix_mu = range_stats['ix_mu']

grid_stats, ix, est_group = grid_search(ADMM_MGL, S, N, p, reg, L1, method= 'eBIC', w2 = W2, latent = latent, mu = mu, ix_mu = ix_mu)


np.linalg.norm(est_group['Theta'] - est_indv['Theta'])/ np.linalg.norm(est_indv['Theta'])

sns.heatmap((est_group['Theta']-Theta).mean(axis=0), square = True, vmin = -.1, vmax = .1, cmap = 'coolwarm')

