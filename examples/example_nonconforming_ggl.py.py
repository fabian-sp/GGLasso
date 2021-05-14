"""
Nonconforming Group Graphical Lasso experiment
===================================================

Example script for Group Graphical Lasso with non-conforming dimension, i.e. some variables exist in some instances but not in all
We use 5 datasets of microbiome gut OTU counts. Not all OTUs are present in all datasets hence we are in the situation above.

We estimate a network of all appearing OTUs with a group sparsity penalty on OTU pairs that appear in multiple datasets.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer

from gglasso.helper.experiment_helper import surface_plot
from gglasso.helper.utils import sparsity, zero_replacement, normalize, log_transform
from gglasso.helper.ext_admm_helper import check_G, consensus

from gglasso.problem import glasso_problem

def load_and_transform(K = 5, min_inst = 2, compute_G = False):
    """

    Parameters
    ----------
    K : int, optional
        DESCRIPTION. The default is 5.
    min_inst : TYPE, optional
        DESCRIPTION. The default is 2.
    compute_G : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    all_csv : dict
        dictionary with transformed sample data for each instance k=1,..,K..
        Transformation includes zero replacement, normalization and log transform.
    S : dict
        dictionary with empirical covariance matrices for each instance k=1,..,K.
    G : array
        bookeeping array needed for the solver.
    ix_location : DataFrame
        DESCRIPTION.
    ix_exist : DataFrame
        DESCRIPTION.
    p : array
        dimensions.
    num_samples : array
        sample sizes.

    """
    all_csv = dict()
    num_csv = K
    num_samples = np.zeros(K, dtype = int)
    p = np.zeros(K, dtype = int)
    
    for num in np.arange(num_csv):      
        file = "../../data/microbiome/OTU_data_" + str(num+1) + ".csv"
        dt = pd.read_csv(file, index_col = 0).sort_index()
        all_csv[num] = dt.copy()
        
        assert (dt.T.dtypes == 'int64').all(), f"instance {num}: {dt.dtypes.unique()}"
    
    # function takes list as input
    ix_exist, ix_location = construct_indexer(list(all_csv.values()))
    
    if compute_G:
        G = create_group_array(ix_exist, ix_location, min_inst)
    else:
        G = None
        
    # finally do clr-transform
    for num in np.arange(num_csv):
        X = all_csv[num]
        X = zero_replacement(X)
        X = normalize(X)
        X = log_transform(X)
        
        all_csv[num] = X.copy()
        # info of dimension and sample size
        p[num] = X.shape[0]
        num_samples[num] = X.shape[1]
    
    # compute covariance matrices
    S = dict()
    for num in np.arange(num_csv):
        S0 = np.cov(all_csv[num].values, bias = True)
        
        # scale covariances to correlations
        scale = np.tile(np.sqrt(np.diag(S0)),(S0.shape[0],1))
        scale = scale.T * scale
        
        S[num] = S0 / scale
        
    return all_csv, S, G, ix_location, ix_exist, p, num_samples


#%% 
# Load the data: we load the OTU counts and do a clr-transform with constant zero-replacement.
# 

K = 4

# load data, transform OTU counts
# the bookeeping array G is created within the function
all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 4, compute_G = True)

check_G(G, p)

#%%
# Dimensionality of the problem.
# In the section above we created two important objects for the non-conforming case:
# ``ix_location`` is a dataframe with every variable (=OTU) as index and each columns is one dataset. It contains the index of the OTU in the respective dataset (``NaN`` when not present).
# ``G`` is a bookeeping array which keeps count of the indices of each group of overlapping variables. It is needed in the solver later on.
#

print("Dimensions p_k: ", p)

print("Sample sizes N_k: ", num_samples)

print("Number of groups found: ", G.shape[1])

print("Indices of the OTUs in each dataset:")
print(ix_location.head())

#%% parameter setup
# We now create the instance of Group Graphical Lasso problem.

reg = 'GGL'

P = glasso_problem(S = S, N = num_samples, reg = "GGL", reg_params = None, latent = True, G = G, do_scaling = True)
print(P)

#%%
# Set the regularization parameter grids and do model selection.
# WARNING: this will run for a while


l1 = np.logspace(0, -2, 5)
mu1 = 2*np.logspace(1, -1, 3)
l2 = np.logspace(-2, -4, 3)
modelselect_params = {'lambda1_range' : l1, 'mu1_range': mu1, 'lambda2_range': l2}

P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1, tol = 1e-8, rtol = 1e-8)

#%% 
# Access the results from model selection.
# 

print("Solution sparsity (ratio of nonzero entries): ")
for k in range(K):
    print(f"Dataset {k}: ", sparsity(P.solution.precision_[k]))


stats = P.modelselect_stats.copy()
stats1 = P.stage1_stats.copy()

# surface plot of eBIC values
fig = surface_plot(stats['L1'], stats['L2'], stats['BIC'])


