"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from microb_helper import load_and_transform

from gglasso.helper.experiment_helper import adjacency_matrix, sparsity
from gglasso.helper.ext_admm_helper import check_G, load_G, save_G, consensus
from gglasso.helper.model_selection import lambda_parametrizer, ebic, surface_plot, map_l_to_w


from gglasso.problem import glasso_problem

#%% load data
K = 4

all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 4, compute_G = True)

#save_G('', G)
#G = load_G('')
check_G(G, p)

#%%

print("Dimensions p_k: ", p)

print("Sample sizes N_k: ", num_samples)

print("Number of groups found: ", G.shape[1])

#%% parameter setup

reg = 'GGL'

l1 = np.logspace(0, -2, 5)
mu1 = np.logspace(1, -1, 3)
w2 = np.logspace(-1, -4, 3)

modelselect_params = {'lambda1_range' : l1, 'mu1_range': mu1, 'w2_range': w2}


P = glasso_problem(S = S, N = num_samples, reg = "GGL", reg_params = None, latent = True, G = G, do_scaling = True)
print(P)

P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1)

#%% evaluation

stats = P.modelselect_stats.copy()

stats1 = P.stage1_stats.copy()



fig = surface_plot(stats['L1'], stats['L2'], stats['BIC'])


