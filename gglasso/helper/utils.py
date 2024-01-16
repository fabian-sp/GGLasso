"""
author: Fabian Schaipp
"""

import numpy as np

from .basic_linalg import adjacency_matrix


def get_K_identity(K, p):
    res = np.zeros((K,p,p))
    for k in np.arange(K):
        res[k,:,:] = np.eye(p)
    
    return res

def sparsity(S):
    """off-diagonal ratio of nonzero entries in S"""
    assert len(S.shape) == 2
    (p,p) = S.shape
    off_nnz = np.count_nonzero(S) - p
    s = off_nnz/(p**2-p)
    return s

def mean_sparsity(S):
    if type(S) == dict:
        s = [sparsity(S[k]) for k in S.keys()]
    elif type(S) == np.ndarray:
        s = [sparsity(S[k,:,:]) for k in range(S.shape[0])]
        
    return np.mean(s)

def hamming_distance(X, Z, t = 1e-10):
    A = adjacency_matrix(X, t=t)
    B = adjacency_matrix(Z, t=t)
    
    return (A+B == 1).sum()
    
def l1norm_od(Theta):
    """
    calculates the off-diagonal l1-norm of a matrix
    """
    (p1,p2) = Theta.shape
    res = 0
    for i in np.arange(p1):
        for j in np.arange(p2):
            if i == j:
                continue
            else:
                res += abs(Theta[i,j])
                
    return res

def deviation(Theta):
    """
    calculates the deviation of subsequent Theta estimates
    deviation = off-diagonal l1 norm
    """
    #tmp = np.roll(Theta, 1, axis = 0)
    (K,p,p) = Theta.shape
    d = np.zeros(K-1)
    for k in np.arange(K-1):
        d[k] = l1norm_od(Theta[k+1,:,:] - Theta[k,:,:]) / l1norm_od(Theta[k,:,:])
        
    return d

#%% functional graphical lasso

def frob_norm_per_block(S, M, off_diag=False):
    (pM, pM) = S.shape
    assert pM%M == 0
    
    p = int(pM/M)
    Y = np.zeros((p,p))
    
    for i in np.arange(p):
        for j in np.arange(start=i, stop=p):
            if j == i:
                if off_diag:
                    Y[i,j] = 0.
                else:
                    Y[i,j] = np.linalg.norm(S[i*M:(i+1)*M,j*M:(j+1)*M])
            else:
                Y[i,j] = np.linalg.norm(S[i*M:(i+1)*M,j*M:(j+1)*M])
                Y[j,i] = Y[i,j]
   
    return Y

def lambda_max_fsgl(S, M):
    """
    computes lambda_max for Funtional Single Graphical Lasso (FSGL)
    
    Let X0 be matrix which is zero on all off-diagonal subblocks
    Idea: for which lambda is zero in the subdifferential of the objective function at X0 
    
    -log det X0  + <S,X0> has derivative S (on the off-diagonal blocks) 
    
    as D[-log det](X0)[Y] = <X0^-1, Y> and X0^-1 is zero on off-diagonal blocks
    
    subdiff of Frobenius norm at 0 is (Frobenius norm) ball of radius 1
    
    --> If lambda >= max_j,l ||S^M_{jl}||_F , then 0 is in subdiff at X0 
    (for each j,l pick U_jl = -1/lambda S_jl from subdiff)
    """
    Y = frob_norm_per_block(S, M, off_diag=True)
    
    return Y.max()

#%% utils for microbiome count data --> clr transform with zero replacement

def geometric_mean(x):
    """
    calculates the geometric mean of a vector
    """
    a = np.log(x)
    return np.exp(a.sum()/len(a))

def zero_replacement(X, c = 0.5):
    """
    replaces zeros with a constant value c
    """
    Z = X.replace(to_replace = 0, value = c)
    return Z

def normalize(X):
    """
    transforms to the simplex
    X should be of a pd.DataFrame of form (p,N)
    """
    return X / X.sum(axis=0)

def log_transform(X):
    """
    log transform, scaled with geometric mean
    X should be a pd.DataFrame of form (p,N)
    """
    g = X.apply(geometric_mean)
    Z = np.log(X / g)
    return Z