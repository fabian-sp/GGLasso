"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
from itertools import combinations



from gglasso.helper.ext_admm_helper import create_group_array

        
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

def load_and_transform(K = 26, min_inst = 5, compute_G = False):
    all_csv = dict()
    all_ix = pd.Index([])
    num_csv = K
    num_samples = np.zeros(K, dtype = int)
    p = np.zeros(K, dtype = int)
    
    for num in np.arange(num_csv):      
        file = "data/slr_data/OTU_data_" + str(num+1) + ".csv"
        dt = pd.read_csv(file, index_col = 0).sort_index()
        all_ix = all_ix.union(dt.index)
        all_csv[num] = dt.copy()
    
    # create info of which location each feature has in each instance
    ix_exist = pd.DataFrame(index = all_ix, columns = np.arange(num_csv)) 
    ix_location = pd.DataFrame(index = all_ix, columns = np.arange(num_csv)) 
    for num in np.arange(num_csv):
        exist = all_ix.isin(all_csv[num].index)
        locations = [all_csv[num].index.get_loc(i) for i in all_ix[exist]]
        ix_exist.loc[:,num] = exist
        ix_location.loc[exist,num] = locations
    
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




            
    






