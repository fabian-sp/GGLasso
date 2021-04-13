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

def construct_trivial_G(p, K):
    """
    this functions construct the array G for the trivial case, i.e. all variables present in all instances
    only needed for checks and testing
    """
    L = int(p*(p-1)/2)
    G = np.zeros((2,L,K), dtype = int)
    for i in np.arange(p):
        for j in np.arange(start = i+1, stop =p):       
            ix = lambda i,j : i*p - int(i*(i+1)/2) + j - 1 - i*1
            G[0, ix(i,j), :] = i
            G[1, ix(i,j), :] = j
            
    return G

def construct_indexer(samples):
    """
    Creates DataFrames which store information which variables exist in which instances (and where they can be found)
    
    Parameters
    ----------
    samples : list
        list of pd.DataFrame, each element should be of shape (n_variables, n_samples).
        The index of the DataFrame should contain unique identifiers (at best integers) for each variable.

    Returns
    -------
    ix_exist : DataFrame
        DESCRIPTION.
    ix_location : DataFrame
        DESCRIPTION.

    """
    K = len(samples)
    
    all_ix = pd.Index([])
    
    for k in np.arange(K):      
        all_ix = all_ix.union(samples[k].index)

    # create info of which location each feature has in each instance
    ix_exist = pd.DataFrame(index = all_ix, columns = np.arange(K)) 
    ix_location = pd.DataFrame(index = all_ix, columns = np.arange(K)) 
    for k in np.arange(K):
        exist = all_ix.isin(samples[k].index)
        locations = [samples[k].index.get_loc(i) for i in all_ix[exist]]
        ix_exist.loc[:,k] = exist
        ix_location.loc[exist,k] = locations
    
    return ix_exist, ix_location

def check_G(G, p):
    """
    function to check a bookkeeping group penalty matrix G
    p: vector of length K with dimensions p_k as entries
    """
    K = G.shape[2]
    
    assert G.dtype == int, "G needs to be an integer array"
    
    assert np.all(G.sum(axis = 2) >= -K), "G has rows with only -1 entries"
    
    assert np.all(((G==-1).sum(axis = 0) == 2) | ((G==-1).sum(axis = 0) == 0)), "Only row or column index specified in some group"
    
    assert np.all((G[0,:,:] + G[1,:,:] == -2) | (G[0,:,:] != G[1,:,:])), "G has entries on the diagonal!"
    
    assert np.all(G >=-1), "No negative indices allowed (only -1 for indicating a missing feature)"
    
    assert np.all(G.max(axis = (0,1)) < p), "indices larger as dimension were found"
    
    assert np.all(G[0,:,:] <= G[1,:,:]), "Only upper diagonal entries should be contained in G"
    
    return

    
def create_group_array(ix_exist, ix_location, min_inst = 2):
    
    (p,K) = ix_exist.shape
    all_ix = ix_exist.index
    
    A = ix_exist.values.astype(int) @ ix_exist.values.astype(int).T
    # diagonal entries are not relevant, A is symmetric and we only need each pair once!
    np.fill_diagonal(A,0)
    A = np.triu(A)
    L = (A >= min_inst).sum()
    all_pairs = np.argwhere(A >= min_inst)
    
    G1 = np.zeros((L,K), dtype = int)
    G2 = np.zeros((L,K), dtype = int)
    bar = 0.1
    
    print("Creation of bookeeping array..." )
    for l in np.arange(L):
        if l/L >= bar:
            print(str(round(bar*100)) + "% finished")
            bar += .1
        p = all_ix[all_pairs[l]]
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
        
            G1[l,:] = tmp1
            G2[l,:] = tmp2
            
    G = np.stack((G1,G2))
    
    return G

def consensus(sol, G):
    """
    nnz: nonzeros per group
    adj: adjacency matrix per group (edge present or not)
    val: value of precision matrix per group, per dataset
    """
    L = G.shape[1]
    #groupsize = (G!=-1).sum(axis=2)[0]
    K = G.shape[2]
    nnz = np.zeros(L)
    
    val =  np.zeros((L,K))
    adj = -1 * np.ones((L,K))
    for l in np.arange(L):
        for k in np.arange(K):
            if G[0,l,k] == -1:
                val[l,k] = np.nan
                continue
            else:
                val[l,k] = sol[k][G[0,l,k], G[1,l,k]]
                nnz[l] += abs(sol[k][G[0,l,k], G[1,l,k]]) >= 1e-5
                adj[l,k] = abs(sol[k][G[0,l,k], G[1,l,k]]) >= 1e-5
                
    return nnz, adj, val 