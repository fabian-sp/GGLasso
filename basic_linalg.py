#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:44:40 2019

@author: fabian
"""

import numpy as np


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
    bdotb = dot(b,b)
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
        
        if np.sqrt(dot(r,r) / bdotb)  <= eps:
            print(f"Reached accuracy after {str(j)} iterations")
            break
        
        beta = dot(r,r)/denom
        
        p = r + beta * p
        
        j += 1
        
    return x



#%%

x_sol = cg_general(lin, dot, b, eps = 1e-5, kwargs = {'A': A})

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



