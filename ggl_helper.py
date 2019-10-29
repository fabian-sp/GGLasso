"""
author: Fabian Schaipp
"""


import pandas as pd
import numpy as np
import scipy as sc

from basic_linalg import t,Gdot,Sdot
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
  # returns the Moreau_Yosida reg. value as well as the proximal map of P
  
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
  
def construct_jacobian_prox_p(X, l1 , l2):
    # each (i,j) entry has a corresponding jacobian which is a KxK matrix
    (K,p,p) = X.shape
    W = np.zeros((K,K,p,p))
    for i in np.arange(p):
        for j in np.arange(p):
            if i == j:
                W[:,:,i,j] = np.eye(K)
            else:
                W[:,:,i,j] = jacobian_prox_phi(X[:,i,j] , l1 , l2) 
    return W
    
def eval_jacobian_prox_p(Y , W):
    # W is the result of construct_jacobian_prox_p
    (K,p,p) = Y.shape
  
    assert W.shape == (K,K,p,p) , "wrong dimensions"
  
    fun = np.zeros((K,p,p))
    for i in np.arange(p):
        for j in np.arange(p):
            fun[:,i,j] = W[:,:,i,j] @ Y[:,i,j]
  
    return fun
  
  
#%%
# functions related to the log determinant
def h(A):
    return - np.log(np.linalg.det(A))

def f(Omega, S):
    return h(Omega).sum() + Gdot(Omega, S)

def phiplus(A, beta, D = np.array([]), Q = np.array([])):
    # D and Q are optional if already precomputed
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)
        print("Eigendecomposition is executed")
    
    phip = lambda d: 0.5 * (np.sqrt(d**2 + 4*beta) + d)
    
    B = Q @ np.diag(phip(D)) @ Q.T
    return B

def phiminus(A, beta , D = np.array([]), Q = np.array([]) ):
    # D and Q are optional if already precomputed
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)
        print("Eigendecomposition is executed")

    phim = lambda d: 0.5 * (np.sqrt(d**2 + 4*beta) - d)
    
    B = Q @ np.diag(phim(D)) @ Q.T
    return B

def moreau_h(A, beta , D = np.array([]), Q = np.array([])):
    # returns the Moreau_Yosida reg. value as well as the proximal map of beta*h
    pp = phiplus(A, beta, D , Q)
    pm = phiminus(A,beta, D , Q)
    psi =  - (beta * np.log (np.linalg.det(pp))) + (0.5 * np.linalg.norm(pm)**2 )
    return psi, pp, pm
  
  
def jacobian_phiplus(A, B, beta, D = np.array([]), Q = np.array([])):
    
    d = A.shape
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)
        print("Eigendecomposition is executed")
    
    phip = lambda d: 0.5 * (np.sqrt(d**2 + 4*beta) + d)
    
    Gamma = np.zeros(d)
    phip_d = phip(D) 
    
    for i in np.arange(d[0]):
        for j in np.arange(d[1]):
            denom = np.sqrt(D[i]**2 + 4* beta) + np.sqrt(D[j]**2 + 4* beta)
            Gamma[i,j] = (phip_d[i] + phip_d[j]) / denom
            
            
    res = Q @ (Gamma * (Q.T @ B @ Q)) @ Q.T
    return res

def construct_gamma(A, beta, D = np.array([]), Q = np.array([])):

    (K,p,p) = A.shape
    Gamma = np.zeros((K,p,p))
    
    if D.shape[0] != A.shape[0]:
        D, Q = np.linalg.eig(A)
        print("Eigendecomposition is executed")
    
    phip = lambda d: 0.5 * (np.sqrt(d**2 + 4*beta) + d)
    
    for k in np.arange(K):
        phip_d = phip(D[k,:]) 
        
        for i in np.arange(p):
            for j in np.arange(p):    
                
                denom = np.sqrt(D[k,i]**2 + 4* beta) + np.sqrt(D[k,j]**2 + 4* beta)
                Gamma[k,i,j] = (phip_d[i] + phip_d[j]) / denom
            
            
    return Gamma

   
def eval_jacobian_phiplus(B, Gamma, Q):
    # Gamma is constructed with construct_gamma
    # Q is the right-eigenvector matrix of the point A
        
    res = Q @ (Gamma * (t(Q) @ B @ Q)) @ t(Q)
    
    assert abs(res - t(res)).max() <= 1e-10
    return res
      

#%%
# functions related to the proximal point algorithm
    
def Phi_t(Omega, Theta, S, Omega_t, Theta_t, sigma_t, lambda1, lambda2):
    
    res = f(Omega, S) + P(Theta, lambda1, lambda2) + 1/(2*sigma_t) * (np.linalg.norm(Omega - Omega_t)**2 + np.linalg.norm(Theta - Theta_t)**2)
    return res

