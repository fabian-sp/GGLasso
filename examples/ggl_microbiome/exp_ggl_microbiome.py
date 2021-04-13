"""
author: Fabian Schaipp

Example script for Group Graphical Lasso with non-conforming dimension, i.e. some variables exist in some instances but not in all
We use 5 datasets of microbiome gut OTU counts. Not all OTUs are present in all datasets hence we are in the situation above.

We estimate a network of all appearing OTUs with a group sparsity penalty on OTU pairs that appear in multiple datasets.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from microb_helper import load_and_transform

from gglasso.helper.experiment_helper import surface_plot
from gglasso.helper.utils import sparsity
from gglasso.helper.ext_admm_helper import check_G, consensus


from gglasso.problem import glasso_problem

#%% load data
K = 4

# load data, transform OTU counts
# the bookeeping array G is created within the function
all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 4, compute_G = True)

check_G(G, p)

#%%

print("Dimensions p_k: ", p)

print("Sample sizes N_k: ", num_samples)

print("Number of groups found: ", G.shape[1])

# ix_location is a dataframe with every variable (=OTU) as index and each columns is one dataset
# it contains the index of the OTU in the respective dataset (NaN when not present)
print("Indices of the OTUs in each dataset:")
print(ix_location.head())


#%% parameter setup

reg = 'GGL'

# determine regularization parameter ranges
l1 = np.logspace(0, -2, 5)
mu1 = 2*np.logspace(1, -1, 3)
w2 = np.logspace(-1, -4, 3)

modelselect_params = {'lambda1_range' : l1, 'mu1_range': mu1, 'w2_range': w2}

# create instance of Graphical Lasso problem
P = glasso_problem(S = S, N = num_samples, reg = "GGL", reg_params = None, latent = True, G = G, do_scaling = True)
print(P)

# do model selection
# WARNING: this will run several minutes
P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1)

#%% evaluation

print("Solution sparsity (ratio of nonzero entries): ")
for k in range(K):
    print(f"Dataset {k}: ", sparsity(P.solution.precision_[k]))


stats = P.modelselect_stats.copy()
stats1 = P.stage1_stats.copy()

# surface plot of eBIC values
fig = surface_plot(stats['L1'], stats['L2'], stats['BIC'])


