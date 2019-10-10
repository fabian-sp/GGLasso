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
    
    return xy.sum() 


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




#%%

A = np.random.normal(size=(2000,2000))
A = A.T @ A

xt = np.random.normal(size= 2000)

b = A @ xt


def lin(x):
    return A@x

def dot(x,y):
    return x@y


#%%

def cg_general(lin, dot, b, eps = 1e-8):
    
    dim = b.shape
    N_iter = len(b)
    x = np.zeros(dim)
    r = b - lin(x)
    p = r.copy()
    j = 0
    
    while j < N_iter :
        
        linp = lin(p)
        alpha = dot(r,r) / dot(p, linp)
        
        x =  x + alpha * p
        denom = dot(r,r)
        r = r -  alpha * linp
        #r = b - linp
        
        if np.sqrt(dot(r,r)) <= eps:
            print("Reached accuracy")
            break
        
        beta = dot(r,r)/denom
        
        p = r + beta * p
        
        j += 1
        
    return x



#%%

x_sol = cg_general(lin, dot, b, eps = 1e-5)

np.linalg.norm(x_sol-xt)


#%%

A = np.random.normal(size=(200,200))
A = A.T @ A + np.eye(A.shape[0])

def lin(X):
    return A @  X

def dot(X, Y):
    return np.trace(X.T @ Y)


Xt = np.random.normal(size=(200,200))
B = lin(Xt)


Xs = cg_general(lin, dot, B, eps = 1e-5)


dot(Xt-Xs, Xt-Xs)


