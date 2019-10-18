#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:44:40 2019

@author: fabian
"""

import numpy as np

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

A = np.random.normal(size=(20,20))
A = A.T @ A + np.eye(A.shape[0])

xt = np.random.normal(size= 20)

b = A @ xt


def lin(x , A):
    return A@x

def dot(x,y):
    return x@y


#%%

def cg_general(lin, dot, b, eps = 1e-8, kwargs = {}):
    """
    This is the CG method for a general selfadjoint linear operator "lin" and a general scalar product "dot"
    
    It solves after x: lin(x) = b
    
    lin: should be a callable where the first argument is the argument of the operator
         other arguments can be handled via kwargs
    dot: should be a callable with two arguments, namely the two points of <X,Y>
    """
    
    dim = b.shape
    N_iter = len(b)
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
        
        if np.sqrt(dot(r,r)) <= eps:
            print(f"Reached accuracy after {str(j)} iterations")
            break
        
        beta = dot(r,r)/denom
        
        p = r + beta * p
        
        j += 1
        
    return x



#%%

x_sol = cg_general(lin, dot, b, eps = 1e-2, kwargs = {'A': A})

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


#%%


def tester(a,b,c):
  return a*b + c

kwargs = {'a':2, 'c':3}

tester(b=3 , **kwargs)

