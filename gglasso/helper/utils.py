"""
author: Fabian Schaipp
"""

import numpy as np

from .basic_linalg import adjacency_matrix


def get_K_identity(K, p):
    res = np.zeros((K,p,p))
    for k in np.arange(K):
        res[k,:,:] = np.eye(p)
    
    return res

def sparsity(S):
    (p,p) = S.shape
    A = adjacency_matrix(S)
    s = A.sum()/(p**2-p)
    return s

def mean_sparsity(S):
    if type(S) == dict:
        s = [sparsity(S[k]) for k in S.keys()]
    elif type(S) == np.ndarray:
        s = [sparsity(S[k,:,:]) for k in range(S.shape[0])]
        
    return np.mean(s)

def hamming_distance(X, Z, t = 1e-10):
    A = adjacency_matrix(X, t=t)
    B = adjacency_matrix(Z, t=t)
    
    return (A+B == 1).sum()
    
def l1norm_od(Theta):
    """
    calculates the off-diagonal l1-norm of a matrix
    """
    (p1,p2) = Theta.shape
    res = 0
    for i in np.arange(p1):
        for j in np.arange(p2):
            if i == j:
                continue
            else:
                res += abs(Theta[i,j])
                
    return res

def deviation(Theta):
    """
    calculates the deviation of subsequent Theta estimates
    deviation = off-diagonal l1 norm
    """
    #tmp = np.roll(Theta, 1, axis = 0)
    (K,p,p) = Theta.shape
    d = np.zeros(K-1)
    for k in np.arange(K-1):
        d[k] = l1norm_od(Theta[k+1,:,:] - Theta[k,:,:]) / l1norm_od(Theta[k,:,:])
        
    return d

#%% utils for microbiome count data --> clr transform with zero replacement

def geometric_mean(x):
    """
    calculates the geometric mean of a vector
    """
    a = np.log(x)
    return np.exp(a.sum()/len(a))

def zero_replacement(X, c = 0.5):
    """
    replaces zeros with a constant value c
    """
    Z = X.replace(to_replace = 0, value = c)
    return Z

def normalize(X):
    """
    transforms to the simplex
    X should be of a pd.DataFrame of form (p,N)
    """
    return X / X.sum(axis=0)

def log_transform(X):
    """
    log transform, scaled with geometric mean
    X should be a pd.DataFrame of form (p,N)
    """
    g = X.apply(geometric_mean)
    Z = np.log(X / g)
    return Z