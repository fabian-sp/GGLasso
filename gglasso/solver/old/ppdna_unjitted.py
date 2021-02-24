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



