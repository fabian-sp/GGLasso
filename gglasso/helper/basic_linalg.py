"""
author: Fabian Schaipp
"""

import numpy as np
from numba import njit

## general functions for the space G

@njit()
def trp(X):   
    # transposes for a block of matrices each single matrix
    # assumes that X is given in the form (K, p, p)
    
    assert len(X.shape) == 3 , "dimension of input has to be 3"
    assert X.shape[1] == X.shape[2] , "third dimension has to be equal to second dimension"
    
    return X.transpose(0,2,1)

@njit()
def Gdot(X, Y):
     # calculates the inner product for X,Y in G = K-fold of symm. matrices in R^p
    (K,p,p) = X.shape
    res = 0
    for k in np.arange(K):
        res += Sdot(X[k,:,:], Y[k,:,:])
    
    return res 


# general functions for the space S
@njit()
def Sdot(X,Y):
    return np.trace(X.T @ Y)

def adjacency_matrix(S, t = 1e-10):
    A = (np.abs(S) >= t).astype(int)
    # do not count diagonal entries as edges
    if len(S.shape) == 3:
        for k in np.arange(S.shape[0]):
            np.fill_diagonal(A[k,:,:], 0)
    else:
        np.fill_diagonal(A, 0)
    return A


def scale_array_by_diagonal(X, d = None):
    """
    scales a 2d-array X with 1/sqrt(d), i.e.
    
    X_ij/sqrt(d_i*d_j)
    in matrix notation: W^-1 @ X @ W^-1 with W^2 = diag(d)
    
    if d = None, use square root diagonal, i.e. W^2 = diag(X)
    see (2.4) in https://fan.princeton.edu/papers/09/Covariance.pdf
    """
    assert len(X.shape) == 2
    if d is None:
        d = np.diag(X)
    else:
        assert len(d) == X.shape[0]
        
    scale = np.tile(np.sqrt(d),(X.shape[0],1))
    scale = scale.T * scale
    
    return X/scale




