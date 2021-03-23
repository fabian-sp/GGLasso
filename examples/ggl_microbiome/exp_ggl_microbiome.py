"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from microb_helper import load_and_transform
from gglasso.solver.ext_admm_solver import ext_ADMM_MGL
from gglasso.solver.single_admm_solver import ADMM_SGL

from gglasso.helper.experiment_helper import adjacency_matrix, sparsity
from gglasso.helper.ext_admm_helper import get_K_identity, check_G, load_G, save_G, consensus
from gglasso.helper.model_selection import lambda_parametrizer, grid_search, K_single_grid, ebic, surface_plot, map_l_to_w


from gglasso.problem import glasso_problem

#%% load data
K = 5

all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 5, compute_G = True)

#save_G('', G)
#G = load_G('')
check_G(G, p)

#%% parameter setup
reg = 'GGL'

l1 = np.logspace(0, -2, 4)
mu1 = np.logspace(1, -1, 3)
w2 = np.logspace(-1, -4, 3)

modelselect_params = {'lambda1_range' : l1, 'mu1_range': mu1, 'w2_range': w2}



P = glasso_problem(S = S, N = num_samples, reg = "GGL", reg_params = None, latent = True, G = G, do_scaling = True)
print(P)

P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1)

#%%

sol2, sol3, range_stats = K_single_grid(S, l1, num_samples, latent = True, mu_range = mu1)

#%%

w2 = np.logspace(-1, -4, 4)
#w2 = np.linspace(0.02, 0.01, 5)

grid_stats, ix, sol1 = grid_search(ext_ADMM_MGL, S, num_samples, p, reg, l1 = l1, method = 'eBIC', w2 = w2, G = G,\
                                   latent = True, mu = mu1, ix_mu = range_stats['ix_mu'])

L1 = grid_stats['L1']
L2 = grid_stats['L2']

W1 = map_l_to_w(L1,L2)[0]
W2 = map_l_to_w(L1,L2)[1]

#surface_plot(L1,L2, grid_stats['BIC'])



#%%

this_mu = mu1[range_stats['ix_mu'][:,range_stats['ix_uniform']]]
#this_mu = mu1[ix_mu[:,ix_uniform]]

Omega_0 = get_K_identity(p)
lambda1 = L1[ix]
lambda2 = lambda_parametrizer(lambda1, w2 = 0.01)
sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True, \
                         latent = True, mu1 = this_mu)

res_multiple2 = sol.copy()

