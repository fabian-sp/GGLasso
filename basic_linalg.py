#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:44:40 2019

@author: fabian
"""

import numpy as np

def t(X):   
    # transposes for a block of matrices each single matrix
    # assumes that X is given in the form (K, p, p)
    
    assert len(A.shape) == 3 , "dimension of input has to be 3"
    assert A.shape[1] == A.shape[2] , "thrid dimension has to be equal to second dimension"
    
    return A.transpose(0,2,1)

def Gdot(X, Y):
    # calculates the inner product for X,Y in G
    assert X.shape == Y.shape
    
    xy = np.trace( np.matmul( t(X), Y) , axis1 = 1, axis2 = 2)
    
    return xy   


#%%
A = np.random.normal( size = (5,20,20))
#A = np.array( [ [[1, 2], [3, 4]] , [[5, 6], [7, 8]] ])

#transpose the single matrices like this
A.transpose(0,2,1)

B = np.matmul( t(A), A)


# eigenvalue decomp works for each matrix 
[M,T] = np.linalg.eig(B)



Gdot(A,A) 

# check if trace equal to sum of eigenvalues
M.sum(axis = 1)
