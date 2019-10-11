# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:57:33 2019

@author: fscha
"""

import pandas as pd
import numpy as np
import scipy as sc

#%%
## general functions for the space G
def t(X):   
    # transposes for a block of matrices each single matrix
    # assumes that X is given in the form (K, p, p)
    
    assert len(X.shape) == 3 , "dimension of input has to be 3"
    assert X.shape[1] == X.shape[2] , "third dimension has to be equal to second dimension"
    
    return X.transpose(0,2,1)

def Gdot(X, Y):
    # calculates the inner product for X,Y in G
    assert X.shape == Y.shape
    
    xy = np.trace( np.matmul( t(X), Y) , axis1 = 1, axis2 = 2)
    
    return xy.sum() 


#%%
# functions related to the GGL regularizer
    
def P(X, l1, l2):
    assert min(l1,l2) > 0, "lambda 1 and lambda2 have to be positive"
    d = X.shape
    res = 0
    for i in np.arange(d[1]):
        for j in np.arange(start = i + 1 , stop = d[2]):
            #print(X[:,i,j])
            res += l1 * np.linalg.norm(X[:,i,j] , 1) + l2 * np.linalg.norm(X[:,i,j] , 2)
    
    # multiply by 2 as we only summed the upper triangular
    return 2 * res

def prox_1norm(v, l): 
    return np.sign(v) * np.maximum(abs(v) - l, 0)
    
def prox_2norm(v,l):
    a = max(np.linalg.norm(v,2) , l)
    return v * (a - l) / a
    
def prox_phi(v, l1, l2):
    assert min(l1,l2) > 0, "lambda 1 and lambda2 have to be positive"
    u = prox_1norm(v, l1)
    return prox_2norm(u,l2)
    
def prox_p(X, l1, l2):
    assert min(l1,l2) > 0, "lambda 1 and lambda2 have to be positive"
    d = X.shape
    M = np.zeros(d)
    for i in np.arange(d[1]):
        for j in np.arange(d[2]):
            if i == j:
                M[:,i,j] = X[:,i,j]
            else:
                M[:,i,j] = prox_phi(X[:,i,j], l1, l2)
    
    assert abs(M - t(M)).max() <= 1e-10
    return M
            
          
def jacobian_projection(v, l):
    
    K = len(v)
    a = np.linalg.norm(v)
    if a <= l:
        g = np.eye(K)
    else:
        g = (l/a) * ( np.eye(K) - (1/a**2) * np.outer(v,v) )
        
    return g
    

def jacobian_2norm(v, l):
    # jacobian of the euclidean norm: v is the vector, l is lambda_2
    K = len(v)
    g = np.eye(K) - jacobian_projection(v, l)
    return g


def jacobian_1norm(v, l):
    
    d = (abs(v) > l).astype(int)
    return np.diag(d)

def jacobian_prox_phi(v , l1 , l2):
    
    u = prox_1norm(v, l1)
    sig = jacobian_projection(u, l2)
    lam = jacobian_1norm(v, l1)
    
    M = (np.eye(len(v)) - sig) @ lam
    assert abs(M - M.T).max() <= 1e-10
    return M

#%%
# testing
v = np.array([1.5,0.3,0])
l = 1.4

jacobian_projection(v,12)


A = np.array( [ [[1, 2], [3, 4]] , [[5, 6], [7, 8]] ])



A = np.random.normal( size = (5,20,20))
A = t(A) @ A









