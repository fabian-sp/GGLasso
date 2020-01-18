"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
from itertools import combinations


def get_K_identity(p):
    
    X = dict()
    K = len(p)
    for k in np.arange(K):
        X[k] = np.eye(p[k])
        
    return X


def load_G(path):
    G1 = np.loadtxt(path + 'G1.txt')
    G2 = np.loadtxt(path + 'G2.txt')

    G = np.stack((G1,G2))
    return G.astype(int)

def save_G(path, G):
    np.savetxt(path + 'G1.txt', G[0,:,:])
    np.savetxt(path + 'G2.txt', G[1,:,:])
    return 

def construct_G(p, K):
    L = int(p*(p-1)/2)
    G = np.zeros((2,L,K), dtype = int)
    for i in np.arange(p):
        for j in np.arange(start = i+1, stop =p):       
            ix = lambda i,j : i*p - int(i*(i+1)/2) + j - 1 - i*1
            G[0, ix(i,j), :] = i
            G[1, ix(i,j), :] = j
            
    return G

def check_G(G, p):
    """
    function to check a bookkeeping group penalty matrix G
    p: vector of length K with dimensions p_k as entries
    """
    K = G.shape[2]
    
    assert G.dtype == int, "G needs to be an integer array"
    
    assert np.all(G.sum(axis = 2) >= -K), "G has rows with only -1 entries"
    
    assert np.all((G[0,:,:] + G[1,:,:] == -2) | (G[0,:,:] != G[1,:,:])), "G has entries on the diagonal!"
    
    assert np.all(G >=-1), "No negative indices allowed (only -1 for indicating a missing feature)"
    
    assert np.all(G.max(axis = (0,1)) < p), "indices larger as dimension were found"
    
    return

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
    

