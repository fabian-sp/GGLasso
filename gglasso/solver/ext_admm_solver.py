"""
author: Fabian Schaipp
"""

import numpy as np
import time
import copy
import warnings

from numba import njit
from numba.typed import List


from gglasso.solver.ggl_helper import phiplus, prox_od_1norm, prox_2norm, prox_rank_norm
from gglasso.helper.ext_admm_helper import check_G


def ext_ADMM_MGL(S, lambda1, lambda2, reg , Omega_0, G,\
                 X0 = None, X1 = None, tol = 1e-5 , rtol = 1e-4, stopping_criterion = 'boyd',\
                 rho= 1., max_iter = 1000, verbose = False, measure = False, latent = False, mu1 = None):
    """
    This is an ADMM algorithm for solving the Group Graphical Lasso problem
    where not all instances have the same number of dimensions, i.e. some variables are present in some instances and not in others.
    A group sparsity penalty is applied to all pairs of variables present in multiple instances.
    
    IMPORTANT: As the arrays are non-conforming in dimensions here, we operate on dictionaries with keys 1,..,K (as int) and each value is a array of shape :math:`(p_k,p_k)`.
    
    If ``latent=False``, this function solves
    
    .. math::
        \min_{\Omega,\Theta,\Lambda} \sum_{k=1}^K - \log \det(\Omega^{(k)}) + \mathrm{Tr}(S^{(k)}\Omega^{(k)}) + \sum_{k=1}^K \lambda_1 ||\Theta^{(k)}||_{1,od} 
                                    + \sum_{l} \lambda_2 \\beta_l ||\Lambda_{[l]}||_2
        
        s.t. \quad \Omega^{(k)} = \Theta^{(k)} \quad  k=1,\dots,K
             
        \quad  \quad  \Lambda^{(k)} = \Theta^{(k)} \quad k=1,\dots,K 
    
    where l indexes the groups of overlapping variables and :math:`\Lambda_{[l]}` is the array of all respective components.
    To account for differing group sizes we multiply with :math:`\\beta_l`, the square root of the group size.
    
    If ``latent=True``, this function solves
    
    .. math::
        \min_{\Omega,\Theta,\Lambda,L} \sum_{k=1}^K - \log \det(\Omega^{(k)}) + \mathrm{Tr}(S^{(k)}\Omega^{(k)}) + \sum_{k=1}^K \lambda_1 ||\Theta^{(k)}||_{1,od} 
        
        + \sum_{l} \lambda_2 \\beta_l ||\Lambda_{[l]}||_2 +\sum_{k=1}^{K} \mu_{1,k} \|L^{(k)}\|_{\star}
        
        s.t. \quad \Omega^{(k)} = \Theta^{(k)} - L^{(k)} \quad  k=1,\dots,K
             
        \quad  \quad  \Lambda^{(k)} = \Theta^{(k)} \quad k=1,\dots,K 
    
    Note:
       * Typically, ``sol['Omega']`` is positive definite and ``sol['Theta']`` is sparse.
       * We use scaled ADMM, i.e. X0 and X1 are the scaled (with 1/rho) dual variables for the equality constraints.  

    Parameters
    ----------
    S : dict 
        empirical covariance matrices. S should have keys 1,..,K (as integers) and S[k] contains the :math:`(p_k,p_k)`-array of the empirical cov. matrix of the k-th instance. 
        Each S[k] needs to be symmetric and positive semidefinite.
    lambda1 : float, positive
        sparsity regularization parameter.
    lambda2 : float, positive
        group sparsity regularization parameter.
    reg : str
        so far only Group Graphical Lasso is available, hence choose 'GGL'.
    Omega_0 : dict
        starting point for the Omega variable. Should be of same form as S. If no better starting point is available, choose
        Omega_0[k] = np.eye(p_k) for k=1,...,K
    G : array
        bookkeeping arrays which contains information where the respective entries for each group can be found.
    X0 : dict, optional
        starting point for the X0 variable. If not specified, it is set to zeros.
    X1 : dict, optional
        starting point for the X1 variable. If not specified, it is set to zeros.
    rho : float, positive, optional
        step size paramater for the augmented Lagrangian in ADMM. The default is 1. Tune this parameter for optimal performance.
    max_iter : int, optional
        maximum number of iterations. The default is 1000.
    tol : float, positive, optional
        tolerance for the primal residual. See "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", Boyd et al. for details.
        The default is 1e-7.
    rtol : float, positive, optional
        tolerance for the dual residual. The default is 1e-4.
    stopping_criterion : str, optional
    
        * 'boyd': Stopping criterion after Boyd et al.
        * 'kkt': KKT residual is chosen as stopping criterion. This is computationally expensive to compute.
        
        The default is 'boyd'.
    verbose : boolean, optional
        verbosity of the solver. The default is False.
    measure : boolean, optional
        turn on/off measurements of runtime per iteration. The default is False.
    latent : boolean, optional
        Solve the GGL problem with or without latent variables (see above for the exact formulations).
        The default is False.
    mu1 : float, positive, optional
        low-rank regularization parameter, possibly different for each instance k=1,..,K. Only needs to be specified if latent=True.

    Returns
    -------
    sol : dict
        contains the solution, i.e. Omega, Theta, X0, X1 (and L if latent=True) after termination. All elements are dictionaries with keys 1,..,K and (p_k,p_k)-arrays as values.
    info : dict
        status and measurement information from the solver.

    
    """
    K = len(S.keys())
    p = np.zeros(K, dtype= int)
    for k in np.arange(K):
        p[k] = S[k].shape[0]
        
    if type(lambda1) == np.float64 or type(lambda1) == float:
        lambda1 = lambda1*np.ones(K)
    if latent:
        if type(mu1) == np.float64 or type(mu1) == float:
             mu1 = mu1*np.ones(K)
            
        assert mu1 is not None
        assert np.all(mu1 > 0)
        
    assert min(lambda1.min(), lambda2) > 0
    assert reg in ['GGL']
   
    check_G(G, p)
    
    assert rho > 0, "ADMM penalization parameter must be positive."
        
    
    # initialize 
    Omega_t = Omega_0.copy()
    Theta_t = Omega_0.copy()
    L_t = dict()
    
    for k in np.arange(K):
        L_t[k] = np.zeros((p[k],p[k]))
    
    # helper and dual variables
    Lambda_t = Omega_0.copy()
    Z_t = dict()

    if X0 is None:
        X0_t = dict()
        for k in np.arange(K):
            X0_t[k] = np.zeros((p[k],p[k]))  
    else:
        X0_t = X0.copy()
        
    if X1 is None:   
        X1_t = dict()
        for k in np.arange(K):
            X1_t[k] = np.zeros((p[k],p[k]))
    else:
        X1_t = X1.copy()
     
     
    runtime = np.zeros(max_iter)
    residual = np.zeros(max_iter)
    status = ''
    
    if verbose:
        print("------------ADMM Algorithm for Multiple Graphical Lasso----------------")

        if stopping_criterion == 'boyd':
            hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s"
            out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
            print(hdr_fmt % ("iter", "r_t", "s_t", "eps_pri", "eps_dual"))
        elif stopping_criterion == 'kkt':
            hdr_fmt = "%4s\t%10s"
            out_fmt = "%4d\t%10.4g"
            print(hdr_fmt % ("iter", "kkt residual"))
            
    ##################################################################
    ### MAIN LOOP STARTS
    ##################################################################
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
 
        # Omega Update
        Omega_t_1 = Omega_t.copy()
        for k in np.arange(K):
            W_t = Theta_t[k] - L_t[k] - X0_t[k] - (1/rho) * S[k]
            eigD, eigQ = np.linalg.eigh(W_t)
            Omega_t[k] = phiplus(beta = 1/rho, D = eigD, Q = eigQ)
        
        # Theta Update
        for k in np.arange(K): 
            V_t = (Omega_t[k] + L_t[k] + X0_t[k] + Lambda_t[k] - X1_t[k]) * 0.5
            Theta_t[k] = prox_od_1norm(V_t, lambda1[k]/(2*rho))
        
        #L Update
        if latent:
            for k in np.arange(K):
                C_t = Theta_t[k] - X0_t[k] - Omega_t[k]
                C_t = (C_t.T + C_t)/2
                eigD, eigQ = np.linalg.eigh(C_t)
                L_t[k] = prox_rank_norm(C_t, mu1[k]/rho, D = eigD, Q = eigQ)
        
        # Lambda Update
        Lambda_t_1 = Lambda_t.copy()
        for k in np.arange(K): 
            Z_t[k] = Theta_t[k] + X1_t[k]
            
        Lambda_t = prox_2norm_G(Z_t, G, lambda2/rho)
        
        # X Update
        for k in np.arange(K):
            X0_t[k] +=  Omega_t[k] - Theta_t[k] + L_t[k]
            X1_t[k] +=  Theta_t[k] - Lambda_t[k]
        
        if measure:
            end = time.time()
            runtime[iter_t] = end-start
        
        # Stopping condition
        if stopping_criterion == 'boyd':
            r_t,s_t,e_pri,e_dual = ADMM_stopping_criterion(Omega_t, Omega_t_1, Theta_t, L_t, Lambda_t, Lambda_t_1, X0_t, X1_t,\
                                                           S, rho, p, tol, rtol, latent)
            
            residual[iter_t] = max(r_t,s_t)
            
            if verbose:
                print(out_fmt % (iter_t,r_t,s_t,e_pri,e_dual))
                
            if (r_t <= e_pri) and  (s_t <= e_dual):
                status = 'optimal'
                break
            
        elif stopping_criterion == 'kkt':
            eta_A = kkt_stopping_criterion(Omega_t, Theta_t, L_t, Lambda_t, dict((k, rho*v) for k,v in X0_t.items()), dict((k, rho*v) for k,v in X1_t.items()),\
                                                S , G, lambda1, lambda2, reg, latent, mu1)
            residual[iter_t] = eta_A
            
            if verbose:
                print(out_fmt % (iter_t,eta_A))
                
            if eta_A <= tol:
                status = 'optimal'
                break
            
        
    ##################################################################
    ### MAIN LOOP FINISHED
    ##################################################################
    
    # retrieve status (partially optimal or max iter)
    if status != 'optimal':
        if stopping_criterion == 'boyd':
            if (r_t <= e_pri):
                status = 'primal optimal'
            elif (s_t <= e_dual):
                status = 'dual optimal'
            else:
                status = 'max iterations reached'
        else:
            status = 'max iterations reached'
        
    print(f"ADMM terminated after {iter_t+1} iterations with status: {status}.")
    
    for k in np.arange(K):
        
        ### CHECK FOR SYMMETRY
        if abs((Omega_t[k]).T - Omega_t[k]).max() > 1e-5:
            warnings.warn(f"Omega variable is not symmetric, largest deviation is {abs((Omega_t[k]).T - Omega_t[k]).max()}.")
        
        if abs((Theta_t[k]).T - Theta_t[k]).max() > 1e-5:
            warnings.warn(f"Theta variable is not symmetric, largest deviation is {abs((Theta_t[k]).T - Theta_t[k]).max()}.")
        
        if abs((L_t[k]).T - L_t[k]).max() > 1e-5:
            warnings.warn(f"L variable is not symmetric, largest deviation is {abs((L_t[k]).T - L_t[k]).max()}.")

        ### CHECK FOR POSDEF
        D = np.linalg.eigvalsh(Theta_t[k]-L_t[k])
        if D.min() <= 1e-5:
            print("WARNING: Theta (Theta-L resp.) may be not positive definite -- increase accuracy!")
                     
        if latent:
            D = np.linalg.eigvalsh(L_t[k])
            if D.min() <= -1e-5:
                print("WARNING: L may be not positive semidefinite -- increase accuracy!")
        
    
    sol = {'Omega': Omega_t, 'Theta': Theta_t, 'L': L_t, 'X0': X0_t, 'X1': X1_t}
    if measure:
        info = {'status': status , 'runtime': runtime[:iter_t+1], 'residual': residual[:iter_t+1]}
    else:
        info = {'status': status}
               
    return sol, info


def ADMM_stopping_criterion(Omega, Omega_t_1, Theta, L, Lambda, Lambda_t_1, X0, X1, S, rho, p, eps_abs, eps_rel, latent=False):
    # X0, X1 are inputed as scaled dual vars., this is accounted for by factor rho in e_dual
    K = len(S.keys())

    if not latent:
        for k in np.arange(K):
            assert np.all(L[k]==0)


    dim = ((p ** 2 + p) / 2).sum()  # number of elements of off-diagonal matrix
    
    D1 = np.sqrt(sum([np.linalg.norm(Omega[k])**2 + np.linalg.norm(Lambda[k])**2 for k in np.arange(K)] ))
    D2 = np.sqrt(sum([np.linalg.norm(Theta[k] - L[k])**2 + np.linalg.norm(Theta[k])**2 for k in np.arange(K)] ))
    D3 = np.sqrt(sum([np.linalg.norm(X0[k])**2 + np.linalg.norm(X1[k])**2 for k in np.arange(K)] ))
    
    e_pri = dim * eps_abs + eps_rel * np.maximum(D1, D2)
    e_dual = dim * eps_abs + eps_rel * rho * D3
    
    
    r = np.sqrt(sum([np.linalg.norm(Omega[k] - Theta[k] + L[k])**2 + np.linalg.norm(Lambda[k] - Theta[k])**2 for k in np.arange(K)] ))
    s = rho * np.sqrt(sum([np.linalg.norm(Omega[k] - Omega_t_1[k])**2 + np.linalg.norm(Lambda[k] - Lambda_t_1[k])**2 for k in np.arange(K)] ))

    return r,s,e_pri,e_dual

def kkt_stopping_criterion(Omega, Theta, L, Lambda, X0, X1, S , G, lambda1, lambda2, reg, latent = False, mu1 = None):
    # X0, X1 are inputed as UNscaled dual variables(!)
    K = len(S.keys())
    
    if not latent:
        for k in np.arange(K):
            assert np.all(L[k]==0)
        
    term1 = np.zeros(K)
    term2 = np.zeros(K)
    term3 = np.zeros(K)
    term4 = np.zeros(K)
    term5 = np.zeros(K)
    term6 = np.zeros(K)
    V = dict()
    
    for k in np.arange(K):
        eigD, eigQ = np.linalg.eigh(Omega[k] - S[k] - X0[k])
        proxk = phiplus(beta = 1, D = eigD, Q = eigQ)
        # primal varibale optimality
        term1[k] = np.linalg.norm(Omega[k] - proxk) / (1 + np.linalg.norm(Omega[k]))
        term2[k] = np.linalg.norm(Theta[k] - prox_od_1norm(Theta[k] + X0[k] - X1[k] , lambda1[k])) / (1 + np.linalg.norm(Theta[k]))
        
        if latent:
            eigD, eigQ = np.linalg.eigh(L[k] - X0[k])
            proxk = prox_rank_norm(L[k] - X0[k], beta = mu1[k], D = eigD, Q = eigQ)
            term3[k] = np.linalg.norm(L[k] - proxk) / (1 + np.linalg.norm(L[k]))
        
        V[k] = Lambda[k] + X1[k]
        
        # equality constraints
        term5[k] = np.linalg.norm(Omega[k] - Theta[k] + L[k]) / (1 + np.linalg.norm(Theta[k]))
        term6[k] = np.linalg.norm(Lambda[k] - Theta[k]) / (1 + np.linalg.norm(Theta[k]))
    
    
    V = prox_2norm_G(V, G, lambda2)
    for k in np.arange(K):
        term4[k] = np.linalg.norm(V[k] - Lambda[k]) / (1 + np.linalg.norm(Lambda[k]))
    
    res = max(np.linalg.norm(term1), np.linalg.norm(term2), np.linalg.norm(term3), np.linalg.norm(term4), np.linalg.norm(term5), np.linalg.norm(term6) )
    return res

def prox_2norm_G(X, G, l2):
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


#%%
# prox operato in case numba version does not work

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