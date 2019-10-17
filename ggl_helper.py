"""
author: Fabian Schaipp
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

# general functions for the space S
    
def Sdot(X,Y):
    return np.trace( X.T @ Y )
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
  
def moreau_P(X, l1, l2):
  # calculates the moreau envelope of P
  # as we need also the prox later it returns both!
  Y = prox_p(X, l1, l2)
  psi = P(Y, l1, l2) + 0.5 * Gdot(X-Y, X-Y) 
 
  return psi, Y           
          
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

def jacobian_prox_p(X, Y ,l1, l2):
    assert X.shape == Y.shape, "argument has not the same shape as evaluation point"
    assert abs(Y - t(Y)).max() <= 1e-10 , "argument Y is not symmetric"
    
    d = X.shape
    W = np.zeros(d)
    for i in np.arange(d[1]):
        for j in np.arange(d[2]):
            if i == j:
                W[:,i,j] = Y[:,i,j]
            else:
                W[:,i,j] = jacobian_prox_phi(X[:,i,j] , l1 , l2) @ Y[:,i,j]
    
    assert Gdot(W, Y) >= 0 , "not a pos. semidef operator"
    return W

#%%
# functions related to the log determinant
    
def h(A):
    return - np.log(np.linalg.det(A))

def phiplus(A, beta, D = np.array([]), Q = np.array([])):
    # D and Q are optional if already precomputed
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)
    
    phip = lambda d: 0.5 * (np.sqrt(d**2 + 4*beta) + d)
    
    B = Q @ np.diag(phip(D)) @ Q.T
    return B

def phiminus(A, beta , D = np.array([]), Q = np.array([]) ):
    # D and Q are optional if already precomputed
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)

    phim = lambda d: 0.5 * (np.sqrt(d**2 + 4*beta) - d)
    
    B = Q @ np.diag(phim(D)) @ Q.T
    return B

def moreau_h(A, beta , D = np.array([]), Q = np.array([])):
    pp = phiplus(A, beta, D , Q)
    pm = phiminus(A,beta, D , Q)
    psi =  - (beta * np.log (np.linalg.det(pp))) + (0.5 * np.linalg.norm(pm)**2 )
    return psi, pp, pm
  
  
def jacobian_phiplus(A, B, beta, D = np.array([]), Q = np.array([])):
    
    d = A.shape
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)
    
    phip = lambda d: 0.5 * (np.sqrt(d**2 + 4*beta) + d)
    
    Gamma = np.zeros(d)
    phip_d = phip(D) 
    
    for i in np.arange(d[0]):
        for j in np.arange(d[1]):
            denom = np.sqrt(D[i]**2 + 4* beta) + np.sqrt(D[j]**2 + 4* beta)
            Gamma[i,j] = (phip_d[i] + phip_d[j]) / denom
            
            
    res = Q @ (Gamma * (Q.T @ B @ Q)) @ Q.T
    return res
    

#%%
# testing
v = np.array([1.5,0.3,0])
l = 1.4

jacobian_projection(v,12)


A = np.array( [ [[1, 2], [3, 4]] , [[5, 6], [7, 8]] ])

A = np.random.normal( size = (10,10))
A = A.T @ A

X = np.random.normal( size = (5,10,10))
X = t(X) @ X

Y = np.random.normal( size = (5,10,10))
Y = t(Y) @ Y






