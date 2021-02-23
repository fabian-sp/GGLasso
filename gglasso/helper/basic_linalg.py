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

def adjacency_matrix(S , t = 1e-5):
    A = (np.abs(S) >= t).astype(int)
    # do not count diagonal entries as edges
    if len(S.shape) == 3:
        for k in np.arange(S.shape[0]):
            np.fill_diagonal(A[k,:,:], 0)
    else:
        np.fill_diagonal(A, 0)
    return A

#############################################################
#### OLD CG METHOD
#### this code is more general, but not jittable 
#### use hessian_Y as lin input
#############################################################

# def cg_general(lin, dot, b, eps = 1e-6, kwargs = {}, verbose = False):
#     """
#     This is the CG method for a general selfadjoint linear operator "lin" and a general scalar product "dot"
    
#     It solves after x: lin(x) = b
    
#     lin: should be a callable where the first argument is the argument of the operator
#          other arguments can be handled via kwargs
#     dot: should be a callable with two arguments, namely the two points of <X,Y>
#     """
    
#     dim = b.shape
#     N_iter = np.array(dim).prod()
#     x = np.zeros(dim)
#     r = b - lin(x, **kwargs)  
#     p = r.copy()
#     j = 0
    
#     while j < N_iter :
        
#         linp = lin(p , **kwargs)
#         alpha = dot(r,r) / dot(p, linp)
        
#         x +=   alpha * p
#         denom = dot(r,r)
#         r -=  alpha * linp
#         #r = b - linp
        
#         if np.sqrt(dot(r,r))  <= eps:
#             if verbose:
#                 print(f"Reached accuracy in iteration {str(j)}")
#             break
        
#         beta = dot(r,r)/denom
#         p = r + beta * p 
#         j += 1
        
#     return x




