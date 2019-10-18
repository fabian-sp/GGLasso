# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:28:50 2019

@author: fabia
"""

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
    
