"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
from itertools import combinations


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

def load_and_transform(K = 26, min_inst = 5):
    all_csv = dict()
    all_ix = pd.Index([])
    num_csv = K
    
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
    
    G = create_group_array(ix_exist, ix_location, min_inst)
    
    # finally do transformation
    for num in np.arange(num_csv):
        X = all_csv[num]
        X = zero_replacement(X)
        X = normalize(X)
        X = log_transform(X)
        
        all_csv[num] = X.copy()
    
    # compute covariance matrices
    S = dict()
    for num in np.arange(num_csv):
        S[num] = np.cov(all_csv[num].values, bias = True)
        
    return all_csv, S, G, ix_location




def create_group_array(ix_exist, ix_location, min_inst = 2):
    
    (p,K) = ix_exist.shape
    all_ix = ix_exist.index
    
    filter_ix = all_ix[ix_exist.sum(axis = 1) >= min_inst]
    n = len(filter_ix)
    print(f"{0.5*(n**2 -n)} possible pairs were found")
    all_pairs = list(combinations(filter_ix,2))

    g1 = list()
    g2 = list()
    
    for p in all_pairs:
        # nonexisting features are marked with -1 
        tmp1 = -1 * np.ones(K, dtype = int)
        tmp2 = -1 * np.ones(K, dtype = int)
        coexist = ix_exist.loc[p[0]] & ix_exist.loc[p[1]]
        if coexist.sum() < 2:
            continue  
        else:
            # if pair exists at at least two instances --> fill the location of the feature into G
            tmp1[coexist] = ix_location.loc[p[0], coexist]
            tmp2[coexist] = ix_location.loc[p[1], coexist]
        
            g1.append(tmp1)
            g2.append(tmp2)
            
    G1 = np.vstack(g1)
    G2 = np.vstack(g2)
    
    G = np.stack((G1,G2))
    
    return G
    
    
    
    

G = create_group_array(ix_exist, ix_location, min_inst = 5)













