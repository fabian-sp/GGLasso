"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
from itertools import combinations



from gglasso.helper.ext_admm_helper import create_group_array
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



def load_tax_data(K=26):
    all_tax = pd.DataFrame()
    num_csv = K
    for num in np.arange(num_csv):      
        file = "data/slr_data/TAX_data_" + str(num+1) + ".csv"
        dt = pd.read_csv(file, index_col = 0).sort_index()  
        all_tax = pd.concat([all_tax, dt])
        
    all_tax = all_tax.loc[~all_tax.index.duplicated(keep='first')].sort_index()
    
    return all_tax
         
    
def assortativity(Theta, all_tax, level = 'Rank2'):
    """ computes assortativity coefficient for a single network
    Theta: DataFrame with precision matrix
    """
        
    # pandas columns may be strings --> convert to int
    Theta.columns = [int(c) for c in Theta.columns]
    A = adjacency_matrix(Theta.values)
    tmp = pd.DataFrame(A, columns = Theta.columns, index = Theta.index)
    
    # overwrite with taxonomic labels
    tmp.index = all_tax.loc[tmp.index, level]
    tmp.columns = all_tax.loc[tmp.columns, level]
    
    # count and create assortativity table
    tmp2 = tmp.reset_index().melt(id_vars = level, var_name = level+'_1')
    res = tmp2.groupby([level, level+'_1'])['value'].sum().unstack(level=0)
    
    # formula: https://igraph.org/r/doc/assortativity.html
    E = res.sum().sum()
    a = res.sum(axis = 0) / E
    b = res.sum(axis = 1) / E
    
    c = np.diag(res).sum() / E
    assort = ( c - (a*b).sum()) / (1- (a*b).sum())

    return assort, res


