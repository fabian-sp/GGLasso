"""
Generates data for numerical experiments in the thesis

Power law network: methodology is inspired by "The joint graphical lasso for inverse covariance estimation across
multiple classes" from Danaher et al.

"""

import numpy as np
import networkx as nx

from .basic_linalg import trp


def power_law_network(p=100, M=10):
    
    nxseed = 2340
    L = int(p/M)
    assert M*L == p
    
    A = np.zeros((p,p))
    Sigma = np.zeros((p,p))
    
    for m in np.arange(M):
    
        G_m = nx.generators.random_graphs.random_powerlaw_tree(n = L, gamma = 2.8, tries = max(5*p,1000), seed = int(nxseed + m))
        A_m = nx.to_numpy_array(G_m)
        
        # generate random numbers for the nonzero entries
        np.random.seed(1234)
        B1 = np.random.uniform(low = .1, high = .4, size = (L,L))
        B2 = np.random.choice(a = [-1,1], p=[.5, .5], size = (L,L))
        
        A_m = A_m * (B1*B2)
        
        A[m*L:(m+1)*L, m*L:(m+1)*L] = A_m
    
    row_sum_od = 1.5 * abs(A).sum(axis = 1)
    # broadcasting in order to divide ROW-wise
    A = A / row_sum_od[:,np.newaxis]
    
    A = .5 * (A + A.T)
    
    # A has 0 on diagonal, fill with 1s
    A = A + np.eye(p)
    assert all(np.diag(A)==1), "Expected 1s on diagonal"
    
    # make sure A is pos def
    D = np.linalg.eigvalsh(A)
    if D.min() < 1e-8:
        A += (0.1+abs(D.min())) * np.eye(p)    
        
    #D = np.linalg.eigvalsh(A)
    #assert D.min() > 0, f"generated matrix A is not positive definite, min EV is {D.min()}"
    
    Ainv = np.linalg.pinv(A, hermitian = True)
    
    # scale by inverse of diagonal and 0.6*1/sqrt(d_ii*d_jj) on off-diag
    d = np.diag(Ainv)
    scale = np.tile(np.sqrt(d),(Ainv.shape[0],1))
    scale = (1/0.6)*(scale.T * scale)
    
    Sigma = Ainv/scale
    np.fill_diagonal(Sigma, 1)
    
     
    assert abs(Sigma.T - Sigma).max() <= 1e-8
    D = np.linalg.eigvalsh(Sigma)
    assert D.min() > 0, "generated matrix Sigma is not positive definite"
         
    return Sigma

def time_varying_power_network(p=100, K=10, M=10):
    """
    generates a power law network. The first block disappears at half-time, while the second block appears
    third block decays exponentially
    p: dimension
    K: number of instances/time-stamps
    M: number of sublocks in each instance, should be greater or equal than 3
    """  
    Sigma = np.zeros((K,p,p))
    
    L = int(p/M)
    assert M*L == p
    assert M >=3
    
    Sigma_0 = power_law_network(p = p, M = M) 
    
    for k in np.arange(K):
        Sigma_k = Sigma_0.copy()

        if k <= K/2:   
            Sigma_k[L:2*L, L:2*L] = np.eye(L)
        else:
            Sigma_k[0:L, 0:L] = np.eye(L)
                  
        Sigma[k,:,:] = Sigma_k
        
    Theta = np.linalg.pinv(Sigma, hermitian = True)
    
    decay = np.exp(-.5 * np.arange(K)) 
    helper = np.ones((K,L,L)) * decay[:,None,None]
    for k in np.arange(K):
        np.fill_diagonal(helper[k,:,:], 1)

    Theta[:,2*L:3*L, 2*L:3*L] *= helper
    
    Sigma, Theta = ensure_sparsity(Sigma, Theta)
    
    return Sigma, Theta
    
def group_power_network(p=100, K=10, M=10):
    """
    generates a power law network. In each single network one block disappears (randomly)
    p: dimension
    K: number of instances/time-stamps
    M: number of sublocks in each instance
    """  
    Sigma = np.zeros((K,p,p))
    
    L = int(p/M)
    assert M*L == p
    
    Sigma_0 = power_law_network(p = p, M = M)
    # contains the number of the block disappearing for each k=1,..,K
    block = np.random.randint(M, size = K)
    
    for k in np.arange(K):    
        Sigma_k = Sigma_0.copy()           
        if K > 1:
            Sigma_k[block[k]*L : (block[k]+1)*L, block[k]*L : (block[k]+1)*L] = np.eye(L)
        
        Sigma[k,:,:] = Sigma_k
            
    Theta = np.linalg.pinv(Sigma, hermitian = True)
    Sigma, Theta = ensure_sparsity(Sigma, Theta)
    
    return Sigma, Theta    

def ensure_sparsity(Sigma, Theta):
    
    Theta[abs(Theta) <= 1e-2] = 0
    
    D = np.linalg.eigvalsh(Theta)
    assert D.min() > 0, "generated matrix Theta is not positive definite"
    
    Sigma = np.linalg.pinv(Theta, hermitian = True)
    
    return Sigma, Theta

    
def sample_covariance_matrix(Sigma, N):
    """
    samples data for a given covariance matrix Sigma (with K layers)
    return: sample covariance matrix S
    """
    assert abs(Sigma - trp(Sigma)).max() <= 1e-10
    (K,p,p) = Sigma.shape

    sample = np.zeros((K,p,N))
    for k in np.arange(K):
        sample[k,:,:] = np.random.multivariate_normal(np.zeros(p), Sigma[k,:,:], N).T

    S = np.zeros((K,p,p))
    for k in np.arange(K):
        # normalize with N --> bias = True
        S[k,:,:] = np.cov(sample[k,:,:], bias = True)
        
    return S,sample


