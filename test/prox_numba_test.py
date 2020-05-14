from numba import njit
import numpy as np
from numba.typed import List
import copy
from gglasso.helper.ext_admm_helper import construct_trivial_G
from gglasso.solver.ext_admm_solver import prox_2norm_G
from gglasso.solver.ggl_helper import prox_2norm

K = 20
p = 200
l2 = .01

X = dict()
for k in np.arange(K):
    tmp = np.random.rand(p,p)
    X[k] = tmp+tmp.T

G = construct_trivial_G(p, K)


#%%
def numba_prox_2norm_G(X, G, l2):
    """
    calculates the proximal operator at points X for the group penalty induced by G
    G: 2xLxK matrix where the -th row contains the (i,j)-index of the element in Theta^k which contains to group l
       if G has a entry -1 no element is contained in the group for this Theta^k
    X: dictionary with X^k at key k, each X[k] is assumed to be symmetric
    """
    assert l2 > 0
    K = len(X.keys())
    for  k in np.arange(K):
        assert abs(X[k] - X[k].T).max() <= 1e-5, "X[k] has to be symmetric"
    
    d = G.shape
    assert d[0] == 2
    assert d[2] == K
    
    group_size = (G[0,:,:] != -1).sum(axis = 1)
    
    tmpX = List()
    for k in np.arange(K):
        tmpX.append(X[k].copy())
    
    X1 = prox_G_inner(G, tmpX, l2, group_size)
                    
    X1 = dict(zip(np.arange(K), X1))
    
    return X1
#@njit
def numba_copyto(A,B):
    
    (d1,d2) = A.shape
    for i in range(d1):
        for j in range(d2):
            if np.isnan(B[i,j]):
                continue
            else:
                A[i,j] = B[i,j]
    return A

@njit
def prox_G_inner(G, X, l2, group_size):
    L = G.shape[1]
    K = G.shape[2]
    
    for l in np.arange(L):
            
        # for each group construct v, calculate prox, and insert the result. Ignore -1 entries of G
        v0 = np.zeros(K)
        
        for k in np.arange(K):
            if G[0,l,k] == -1:
                v0[k] = np.nan
            else:
                v0[k] = X[k][G[0,l,k], G[1,l,k]]
        
        
        v = v0[~np.isnan(v0)]
        # scale with square root of the group size
        lam = l2 * np.sqrt(group_size[l])
        a = max(np.sqrt((v**2).sum()), lam)
        z0 = v * (a - lam) / a
    
        v0[~np.isnan(v0)] = z0
        
        for k in np.arange(K):
            if G[0,l,k] == -1:
                continue
            else:
                X[k][G[0,l,k], G[1,l,k]] = v0[k]
                # lower triangular
                X[k][G[1,l,k], G[0,l,k]] = v0[k]
        
    return X

def true_prox(X, l2):
    
    (K,p,p) = X.shape
    M = np.zeros((K,p,p))
    for i in np.arange(p):
        for j in np.arange(p):
            if i == j:
                M[:,i,j] = X[:,i,j]
            else:
                M[:,i,j] = prox_2norm(X[:,i,j], np.sqrt(K)*l2)
    
    return M

#%%
    
res1  = numba_prox_2norm_G(X, G, l2)
res1 = np.stack(res1.values())

res2  = prox_2norm_G(X, G, l2)
res2 = np.stack(res2.values())

res3 = true_prox(np.stack(X.values()), l2)





print(np.linalg.norm(res3-res1))
