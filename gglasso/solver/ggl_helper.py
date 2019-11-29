"""
author: Fabian Schaipp
"""

import numpy as np
from tick.prox import ProxTV
from numba import jit

from gglasso.helper.basic_linalg import trp,Gdot,Sdot

# functions specifically related to the GGL regularizer
def prox_1norm(v, l): 
    return np.sign(v) * np.maximum(abs(v) - l, 0)
    
def prox_2norm(v,l):
    a = max(np.linalg.norm(v,2) , l)
    return v * (a - l) / a

def prox_phi_ggl(v, l1, l2):
    u = prox_1norm(v, l1)
    return prox_2norm(u,l2)


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


def jacobian_prox_phi_ggl(v , l1 , l2):
    u = prox_1norm(v, l1)
    sig = jacobian_projection(u, l2)
    lam = jacobian_1norm(v, l1)
    
    M = (np.eye(len(v)) - sig) @ lam
    assert abs(M - M.T).max() <= 1e-10
    return M

# functions specifically related to the FGL regularizer
def construct_B(K):
    dd = np.eye(K)
    ld = - np.tri(K, k = -1) + np.tri(K, k = -2) 
    
    B = dd+ld
    B = np.delete(B, 0, axis = 0)
    # this is the left-inverse of B.T, is needed to reconstruct the dual solution z_lambda
    Binv = np.linalg.pinv(B.T)
    return B, Binv


def prox_tv(v,l):
    #a = condat_method(v,l)
    a = ProxTV(l).call(np.ascontiguousarray(v))
    return a

def prox_phi_fgl(v, l1, l2):
    res = prox_1norm(prox_tv(v,l2) , l1)
    return res


def jacobian_tv(v,l):   
    K = len(v)   
    B, Binv = construct_B(K)
    
    x_l2 = prox_tv(v,l)   
    z_l2 = Binv @ (v - x_l2)
    
    ind1 = (abs(abs(z_l2) - l) <= 1e-10)    
    #ind2 = (B@x_l2 != 0)
    
    Sigma = np.diag(1-ind1.astype(int))   
    P_hat = np.linalg.pinv(Sigma @ B@ B.T @ Sigma , hermitian = True)
    P = np.eye(K) - B.T @ P_hat @ B
    return P 

   
def jacobian_prox_phi_fgl(v , l1 , l2): 
    x = prox_tv(v,l2)
    P = jacobian_tv(v,l2)
    Theta = jacobian_1norm(x, l1)
    
    return Theta @ P

# general functions related to the regularizer P
    
def P_val(X, l1, l2, reg):
    assert min(l1,l2) > 0, "lambda 1 and lambda2 have to be positive"
    d = X.shape
    res = 0
    for i in np.arange(d[1]):
        for j in np.arange(start = i + 1 , stop = d[2]):
            if reg == 'GGL':
                res += l1 * np.linalg.norm(X[:,i,j] , 1) + l2 * np.linalg.norm(X[:,i,j] , 2)
            elif reg == 'FGL':
                res += l1 * np.linalg.norm(X[:,i,j] , 1) + l2 * np.linalg.norm(X[1:,i,j] - X[:-1,i,j] , 1)
                
    # multiply by 2 as we only summed the upper triangular
    return 2 * res
    
def prox_phi(v, l1, l2, reg):
    assert min(l1,l2) > 0, "lambda 1 and lambda2 have to be positive"
    assert reg in ['GGL', 'FGL']
    
    if reg == 'GGL':
        res = prox_phi_ggl(v, l1, l2)
    elif reg == 'FGL':
        res = prox_phi_fgl(v, l1, l2)
    return res
    
def prox_p(X, l1, l2, reg):
    assert min(l1,l2) > 0, "lambda 1 and lambda2 have to be positive"
    d = X.shape
    M = np.zeros(d)
    for i in np.arange(d[1]):
        for j in np.arange(d[2]):
            if i == j:
                M[:,i,j] = X[:,i,j]
            else:
                M[:,i,j] = prox_phi(X[:,i,j], l1, l2 , reg)
    
    assert abs(M - trp(M)).max() <= 1e-10
    return M
  
def moreau_P(X, l1, l2, reg):
  # returns the Moreau_Yosida reg. value as well as the proximal map of P
  Y = prox_p(X, l1, l2, reg)
  psi = P_val(Y, l1, l2, reg) + 0.5 * Gdot(X-Y, X-Y) 
 
  return psi, Y           
          

def jacobian_prox_phi(v , l1 , l2, reg):
    assert reg in ['GGL', 'FGL']
    
    if reg == 'GGL':
        res = jacobian_prox_phi_ggl(v , l1 , l2)
    elif reg == 'FGL':
        res = jacobian_prox_phi_fgl(v , l1 , l2)
        
    return res


def construct_jacobian_prox_p(X, l1 , l2, reg):
    # each (i,j) entry has a corresponding jacobian which is a KxK matrix
    (K,p,p) = X.shape
    W = np.zeros((K,K,p,p))
    for i in np.arange(p):
        for j in np.arange(p):
            if i == j:
                W[:,:,i,j] = np.eye(K)
            else:
                W[:,:,i,j] = jacobian_prox_phi(X[:,i,j] , l1 , l2, reg) 
    return W

@jit(nopython=True)
def eval_jacobian_prox_p(Y , W):
    # W is the result of construct_jacobian_prox_p
    (K,p,p) = Y.shape
  
    assert W.shape == (K,K,p,p)
  
    fun = np.zeros((K,p,p))
    for i in np.arange(p):
        for j in np.arange(p):
            fun[:,i,j] = W[:,:,i,j] @ Y[:,i,j]
  
    return fun
  
# functions related to the log determinant
def h(A):
    return - np.log(np.linalg.det(A))

def f(Omega, S):
    return h(Omega).sum() + Gdot(Omega, S)

@jit(nopython=True)
def phip(d, beta):
    return 0.5 * (np.sqrt(d**2 + 4*beta) + d)

@jit(nopython=True)
def phim(d, beta):
    return 0.5 * (np.sqrt(d**2 + 4*beta) - d)

def phiplus(A, beta, D = np.array([]), Q = np.array([])):
    # D and Q are optional if already precomputed
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)
        print("Single eigendecomposition is executed in phiplus")
    
    B = Q @ np.diag(phip(D,beta)) @ Q.T
    return B

def phiminus(A, beta , D = np.array([]), Q = np.array([]) ):
    # D and Q are optional if already precomputed
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)
        print("Single eigendecomposition is executed in phiminus")
    
    B = Q @ np.diag(phim(D,beta)) @ Q.T
    return B

def moreau_h(A, beta , D = np.array([]), Q = np.array([])):
    # returns the Moreau_Yosida reg. value as well as the proximal map of beta*h
    
    pp = phiplus(A, beta, D , Q)
    pm = phiminus(A,beta, D , Q)
    psi =  - (beta * np.log (np.linalg.det(pp))) + (0.5 * np.linalg.norm(pm)**2 )
    return psi, pp, pm


@jit(nopython=True)
def construct_gamma(A, beta, D = np.array([]), Q = np.array([])):
    (K,p,p) = A.shape
    Gamma = np.zeros((K,p,p))
    
    if D.shape[0] != A.shape[0]:
        raise KeyError
    
    for k in np.arange(K):
        phip_d = phip(D[k,:] , beta) 
        
        for i in np.arange(p):
            for j in np.arange(p):    
                
                denom = np.sqrt(D[k,i]**2 + 4* beta) + np.sqrt(D[k,j]**2 + 4* beta)
                Gamma[k,i,j] = (phip_d[i] + phip_d[j]) / denom
                      
    return Gamma

def eval_jacobian_phiplus(B, Gamma, Q):
    # Gamma is constructed with construct_gamma
    # Q is the right-eigenvector matrix of the point A
        
    res = Q @ (Gamma * (trp(Q) @ B @ Q)) @ trp(Q)
    
    assert abs(res - trp(res)).max() <= 1e-10
    return res
      

# functions related to the proximal point algorithm
    
def Phi_t(Omega, Theta, S, Omega_t, Theta_t, sigma_t, lambda1, lambda2, reg):
    res = f(Omega, S) + P_val(Theta, lambda1, lambda2, reg) + 1/(2*sigma_t) * (np.linalg.norm(Omega - Omega_t)**2 + np.linalg.norm(Theta - Theta_t)**2)
    return res

def hessian_Y(D , Gamma, eigQ, W, sigma_t):
    """
    this is the linear operator for the CG method
    argument is D
    Gamma and W are constructed beforehand in order to evaluate more efficiently
    """
    tmp1 = eval_jacobian_phiplus( D, Gamma, eigQ)
    tmp2 = eval_jacobian_prox_p( D , W)

    res = - sigma_t * (tmp1 + tmp2)
    return res


def Y_t( X, Omega_t, Theta_t, S, lambda1, lambda2, sigma_t, reg):
    assert min(lambda1, lambda2, sigma_t) > 0 , "at least one parameter is not positive"
    assert X.shape[1] == X.shape[2], "dimensions are not as expected"
  
    (K,p,p) = X.shape
  
    W_t = Omega_t - (sigma_t * (S + X))  
    V_t = Theta_t + (sigma_t * X)

    eigD, eigQ = np.linalg.eig(W_t)
    #print("Eigendecomposition is executed in Y_t")
  
    grad1 = np.zeros((K,p,p))
    term1 = 0
    for k in np.arange(K):
        Psi_h, proxh, _ = moreau_h(W_t[k,:,:] , sigma_t, D = eigD[k,:] , Q = eigQ[k,:,:] )
        term1 += (1/sigma_t) * Psi_h
        grad1[k,:,:] = proxh
    
    term2 = - 1/(2*sigma_t) * ( Gdot(W_t, W_t) + Gdot(V_t, V_t))
    term3 = 1/(2*sigma_t) * (  Gdot(Omega_t, Omega_t)  +  Gdot(Theta_t, Theta_t)   )  
  
    Psi_P , U = moreau_P(V_t, sigma_t * lambda1, sigma_t*lambda2, reg)  
    term4 = (1/sigma_t) * Psi_P
  
    fun = term1 + term2 + term3 + term4
    grad = grad1 - U
  
    return fun, grad
