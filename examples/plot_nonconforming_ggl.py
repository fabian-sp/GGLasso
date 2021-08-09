"""
Nonconforming Group Graphical Lasso experiment
===================================================

Example script for Group Graphical Lasso with non-conforming dimension, i.e. some variables exist in some instances but not in all.
We generate one underlying precision matrix and then drop one block of variables in each instance.

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer
from gglasso.helper.utils import sparsity
from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.helper.ext_admm_helper import check_G, consensus

from gglasso.problem import glasso_problem

K = 4
p = 50
M = 10
B = int(p/M)
N = 100

#%% 
# Generating the data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We generate one precision matrix, sample observations, and finally drop one block of variables in each instance.
# It is important that the observations for each instance is a ``DataFrame`` of shape ``(p_k,N_k)`` where the index has unique ids for each variable.
#

p_arr = (p-B)*np.ones(K, dtype = int)
num_samples = N*np.ones(K, dtype = int)

Sigma, Theta = generate_precision_matrix(p=p, M=M, style = 'powerlaw', gamma = 2.8, prob = 0.1, nxseed = 3456)

all_obs = dict()
S = dict()
for k in np.arange(K):
    
    _, obs = sample_covariance_matrix(Sigma, N)
    
    # drop the k-th block starting from the end
    all_obs[k] = pd.DataFrame(obs).drop(np.arange(p-(k+1)*B, p-k*B), axis = 0)
    S[k] = np.cov(all_obs[k], bias = True)


#%%
# Creating the groups
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this section, we create two important objects for the non-conforming case using functionalities of ``GGLasso``:
#
# ``ix_location`` is a dataframe with every variable as index and each columns is one dataset. It contains the index of the variable in the respective instance (``NaN`` when not present).
#
# ``G`` is a bookeeping array which keeps count of the indices of each group of overlapping variables. It is needed in the solver later on.
#
# **Important:** We only consider pairs of variables which appear at least in 3 instances here!

ix_exist, ix_location = construct_indexer(list(all_obs.values()))

G = create_group_array(ix_exist, ix_location, min_inst = K-1)

check_G(G, p)

print("Dimensions p_k: ", p_arr)

print("Sample sizes N_k: ", num_samples)

print("Number of groups found: ", G.shape[1])

#%%
# Visualizing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We visualize the case of non-conforming variables by plotting the given empirical covariance matrices.
# Missing variable observations are in **white**.
#

fig, axs = plt.subplots(2,2, figsize = (8,8))
for k in range(K):
    ind = ix_exist.index[ix_exist.loc[:,k]]
    S_k = pd.DataFrame(S[k], index = ind, columns = ind)
    
    # extend matrix by nonexistent variables
    S_k = S_k.reindex(columns = ix_exist.index, index = ix_exist.index)
    
    ax = axs.ravel()[k]
    sns.heatmap(S_k, ax = ax, cmap = plt.cm.coolwarm, linewidth = 0.005, linecolor = 'lightgrey',\
                cbar = False, vmin = -.5, vmax = .5, xticklabels = [], yticklabels = [])
    ax.set_title(f"Empirical covariance, instance {k}")
    
#%% 
# Defining the GGL problem
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We now create the instance of Group Graphical Lasso problem. As we are in the non-conforming case, we need to spcify the array ``G`` which we created before.

reg = 'GGL'

P = glasso_problem(S = S, N = num_samples, reg = "GGL", reg_params = None, latent = True, G = G, do_scaling = True)
print(P)

#%%
# Model selection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Set the regularization parameter grids and do model selection.

l1 =    np.logspace(1,-1,5)
mu1 =   np.logspace(1,-1,3)
l2 =    np.logspace(0,-2,4)

modelselect_params = {'lambda1_range' : l1, 'mu1_range': mu1, 'lambda2_range': l2}

P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1, tol = 1e-7, rtol = 1e-7)

print(P.reg_params)

#%% 
# Evaluation of results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We print the ratio of non-zero entries per instance.
# We plot the distribution of non-zero entries per group. 
# With group sparsity regularization we aim for many groups with either no non-zero or many non-zero entries per group.
#

print("Solution sparsity (ratio of nonzero entries): ")

for k in np.arange(K):
    print(f"Instance {k}: ", sparsity(P.solution.precision_[k]))


stats = P.modelselect_stats.copy()
nnz,_,_ = consensus(P.solution.precision_, G)

fig, ax = plt.subplots()
sns.histplot(nnz, discrete = True, ax = ax)
ax.set_yscale('log')
ax.set_title('Nonzero entries per group')
