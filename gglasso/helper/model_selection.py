"""
author: Fabian Schaipp
"""

import numpy as np

from .basic_linalg import Sdot, adjacency_matrix


def aic(S,Theta, N):
    """
    AIC information criterion after Danaher et al.
    excludes the diagonal
    """
    (K,p,p) = S.shape
    
    A = adjacency_matrix(Theta , t = 1e-5)
    nonzero_count = A.sum(axis=(1,2))/2
    aic = 0
    for k in np.arange(K):
        aic += N*Sdot(S[k,:,:], Theta[k,:,:]) - N*np.log(np.linalg.det(Theta[k,:,:])) + 2*nonzero_count[k]
        
    return aic

def ebic(S, Theta, N, gamma = 0.5):
    """
    extended BIC after Drton et al.
    """
    (K,p,p) = S.shape
    
    A = adjacency_matrix(Theta , t = 1e-5)
    nonzero_count = A.sum(axis=(1,2))/2
    
    bic = 0
    for k in np.arange(K):
        bic += N*Sdot(S[k,:,:], Theta[k,:,:]) - N*np.log(np.linalg.det(Theta[k,:,:])) + nonzero_count[k] * (np.log(N)+ 4*np.log(p)*gamma)
    
    return bic