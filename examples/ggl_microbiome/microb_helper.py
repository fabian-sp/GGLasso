"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
from itertools import combinations

from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer
from gglasso.helper.basic_linalg import Sdot, adjacency_matrix
        
def geometric_mean(x):
    a = np.log(x)
    return np.exp(a.sum()/len(a))
    
def zero_replacement(X):  
    Z = X.replace(to_replace = 0, value = 0.5)
    return Z

def normalize(X):
    return X / X.sum(axis=0)

def log_transform(X):
    g = X.apply(geometric_mean)
    Z = np.log(X / g)
    return Z

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
        
    # finally do transformation
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


         
    



