#%%
# prox operator in case numba version does not work

# def prox_2norm_G(X, G, l2):
#     """
#     calculates the proximal operator at points X for the group penalty induced by G
#     G: 2xLxK matrix where the -th row contains the (i,j)-index of the element in Theta^k which contains to group l
#        if G has a entry -1 no element is contained in the group for this Theta^k
#     X: dictionary with X^k at key k, each X[k] is assumed to be symmetric
#     """
#     assert l2 > 0
#     K = len(X.keys())
#     for  k in np.arange(K):
#         assert abs(X[k] - X[k].T).max() <= 1e-5, "X[k] has to be symmetric"
    
#     d = G.shape
#     assert d[0] == 2
#     assert d[2] == K
#     L = d[1]
    
#     X1 = copy.deepcopy(X)
#     group_size = (G[0,:,:] != -1).sum(axis = 1)
    
#     for l in np.arange(L):
#         # for each group construct v, calculate prox, and insert the result. Ignore -1 entries of G
#         v0 = np.zeros(K)
#         for k in np.arange(K):
#             if G[0,l,k] == -1:
#                 v0[k] = np.nan
#             else:
#                 v0[k] = X[k][G[0,l,k], G[1,l,k]]
        
#         v = v0[~np.isnan(v0)]
#         # scale with square root of the group size
#         z0 = prox_2norm(v,l2 * np.sqrt(group_size[l]))
#         v0[~np.isnan(v0)] = z0
        
#         for k in np.arange(K):
#             if G[0,l,k] == -1:
#                 continue
#             else:
#                 X1[k][G[0,l,k], G[1,l,k]] = v0[k]
#                 # lower triangular
#                 X1[k][G[1,l,k], G[0,l,k]] = v0[k]
             
#     return X1

############################
## some old snippets befor jitting
############################

# def OLD_prox_p(X, l1, l2, reg):
#     assert min(l1,l2) > 0, "lambda 1 and lambda2 have to be positive"
#     (K,p,p) = X.shape
#     M = np.zeros((K,p,p))
#     for i in np.arange(p):
#         for j in np.arange(p):
#             if i == j:
#                 M[:,i,j] = X[:,i,j]
#             else:
#                 M[:,i,j] = prox_phi(X[:,i,j], l1, l2 , reg)
    
#     assert np.abs(M - trp(M)).max() <= 1e-5, f"symmetry failed by  {abs(M - trp(M)).max()}"
#     return M

# def construct_B(K):
#     dd = np.eye(K)
#     ld = - np.tri(K, k = -1) + np.tri(K, k = -2) 
    
#     B = dd+ld
#     B = B[1:,:]
#     # this is the left-inverse of B.T, is needed to reconstruct the dual solution z_lambda
#     Binv = np.linalg.pinv(B.T)
#     return B, Binv


# def jacobian_tv(v,l):   
#     K = len(v)   
#     B, Binv = construct_B(K)
    
#     x_l2 = prox_tv(v,l)   
#     z_l2 = Binv @ (v - x_l2)
    
#     ind1 = (np.abs(np.abs(z_l2) - l) <= 1e-10)    
    
#     Sigma = np.diag(1-ind1.astype(int))   
#     P_hat = np.linalg.pinv(Sigma @ B@ B.T @ Sigma , hermitian = True)
#     P = np.eye(K) - B.T @ P_hat @ B
#     return P 

#############################################################
#### OLD CG METHOD
#### this code is more general, but not jittable 
#### use hessian_Y as lin input
#############################################################

# def cg_general(lin, dot, b, eps = 1e-6, kwargs = {}, verbose = False):
#     """
#     This is the CG method for a general selfadjoint linear operator "lin" and a general scalar product "dot"
    
#     It solves after x: lin(x) = b
    
#     lin: should be a callable where the first argument is the argument of the operator
#          other arguments can be handled via kwargs
#     dot: should be a callable with two arguments, namely the two points of <X,Y>
#     """
    
#     dim = b.shape
#     N_iter = np.array(dim).prod()
#     x = np.zeros(dim)
#     r = b - lin(x, **kwargs)  
#     p = r.copy()
#     j = 0
    
#     while j < N_iter :
        
#         linp = lin(p , **kwargs)
#         alpha = dot(r,r) / dot(p, linp)
        
#         x +=   alpha * p
#         denom = dot(r,r)
#         r -=  alpha * linp
#         #r = b - linp
        
#         if np.sqrt(dot(r,r))  <= eps:
#             if verbose:
#                 print(f"Reached accuracy in iteration {str(j)}")
#             break
        
#         beta = dot(r,r)/denom
#         p = r + beta * p 
#         j += 1
        
#     return x



