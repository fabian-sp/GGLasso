#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:44:40 2019

@author: fabian
"""

import numpy as np


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

# general functions for the space S
    
def Sdot(X,Y):
    return np.trace( X.T @ Y )
#%%
def cg_general(lin, dot, b, eps = 1e-6, kwargs = {}):
    """
    This is the CG method for a general selfadjoint linear operator "lin" and a general scalar product "dot"
    
    It solves after x: lin(x) = b
    
    lin: should be a callable where the first argument is the argument of the operator
         other arguments can be handled via kwargs
    dot: should be a callable with two arguments, namely the two points of <X,Y>
    """
    
    dim = b.shape
    #N_iter = len(b)
    N_iter = np.array(dim).prod()
    x = np.zeros(dim)
    r = b - lin(x, **kwargs)
    p = r.copy()
    j = 0
    
    while j < N_iter :
        
        linp = lin(p , **kwargs)
        alpha = dot(r,r) / dot(p, linp)
        
        x +=   alpha * p
        denom = dot(r,r)
        r -=  alpha * linp
        #r = b - linp
        
        if np.sqrt(dot(r,r))  <= eps:
            print(f"Reached accuracy in iteration {str(j)}")
            break
        
        beta = dot(r,r)/denom
        
        p = r + beta * p
        
        j += 1
        
    return x







