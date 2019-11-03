# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:28:50 2019

@author: fabia
"""

#%%
# miscalleneous from ggl_helper

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

def jacobian_phiplus(A, B, beta, D = np.array([]), Q = np.array([])):
    
    d = A.shape
    if len(D) != A.shape[0]:
        D, Q = np.linalg.eig(A)
        print("Single eigendecomposition is executed in jacobian_phiplus")
    
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

class LinearOperator:
  def __init__(self, W, V, D, lambda1, lambda2, sigma_t):
    self.D = D
    self.W = W
    self.V = V
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.sigma_t = sigma_t
    self.K = W.shape[0]
    self.p = W.shape[1]
    self.Gamma = Gamma_constructor(W,sigma_t)
    
  def construct(self):
    Gamma = np.zeros((self.K, self.p, self.p))
    for k in np.arange(self.K):
      Gamma[k,:,:] = Gamma_constructor(self.W[k,:,:] , self.sigma_t)
    


#def construct_gamma(A, beta, D = np.array([]), Q = np.array([])):
#    
#    d = A.shape
#    if len(D) != A.shape[0]:
#        D, Q = np.linalg.eig(A)
#    
#    phip = lambda d: 0.5 * (np.sqrt(d**2 + 4*beta) + d)
#    
#    Gamma = np.zeros(d)
#    phip_d = phip(D) 
#    
#    for i in np.arange(d[0]):
#        for j in np.arange(d[1]):
#            denom = np.sqrt(D[i]**2 + 4* beta) + np.sqrt(D[j]**2 + 4* beta)
#            Gamma[i,j] = (phip_d[i] + phip_d[j]) / denom
#            
#            
#    return Gamma

#%%

#x_sol = cg_general(lin, dot, b, eps = 1e-5, kwargs = {'A': A})
#
#np.linalg.norm(x_sol-xt)
#
#
##%%
#
#A = np.random.normal(size=(200,200))
#A = A.T @ A + np.eye(A.shape[0])
#
#def lin(X):
#    return A @  X
#
#def dot(X, Y):
#    return np.trace(X.T @ Y)
#
#
#Xt = np.random.normal(size=(200,200))
#B = lin(Xt)
#
#
#Xs = cg_general(lin, dot, B, eps = 1e-5)
#
#
#dot(Xt-Xs, Xt-Xs)
#
#A = np.array( [ [[1, 2], [3, 4]] , [[5, 6], [7, 8]] ])