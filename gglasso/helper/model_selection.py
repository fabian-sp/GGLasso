"""
author: Fabian Schaipp
"""

import numpy as np

from .basic_linalg import Sdot, adjacency_matrix
from .utils import mean_sparsity, sparsity

from .utils import get_K_identity as id_array
from .ext_admm_helper import get_K_identity as id_dict
from ..solver.single_admm_solver import ADMM_SGL, block_SGL


      
def lambda_parametrizer(l1 = 0.05, w2 = 0.5):
    """transforms given l1 and w2 into the respective l2"""
    a = 1/np.sqrt(2)
    l2 = (w2*l1)/(a*(1-w2))

    return l2

def map_l_to_w(l1, l2):
    w1 = l1 + (1/np.sqrt(2)) *l2
    w2 = l2/(w1*np.sqrt(2))
    
    return (w1,w2)
    
def lambda_grid(l1, l2 = None, w2 = None):
    """
    l1, l2, w2: values for the grid
    either l2 or w2 has to be spcified
    idea: the grid goes from higher to smaller values when going down/right
    """   
    
    assert np.all(l2!=None) | np.all(w2!=None), "Either a range of lambda2 or w2 values have to be specified"
    if np.all(w2!=None):
        l1grid, w2grid = np.meshgrid(l1,w2)
        L2 = lambda_parametrizer(l1grid, w2grid)
        L1 = l1grid.copy()
    elif np.all(l2!=None):
        L1, L2 = np.meshgrid(l1,l2)
        w2 = None
        
    return L1.squeeze(), L2.squeeze(), w2

def grid_search(solver, S, N, p, reg, l1, l2 = None, w2 = None, method= 'eBIC', gamma = 0.3, \
                G = None, latent = False, mu_range = None, ix_mu = None, tol = 1e-7, rtol = 1e-7, verbose = False):
    """
    method for doing model selection for MGL problems using grid search and AIC/eBIC
    parameters to select: lambda1 (sparsity), lambda2 (group sparsity or total variation)
    
    In the grid lambda1 changes over columns, lambda2 over the rows.
    The grid is ran columnwise, i.e. hold l1 constant and change l2.
    
    
    Parameters
    ----------
    solver : solver method 
        DESCRIPTION.
    S : array of shape (K,p,p) or dict
        empirical covariance matrices.
    N : array
        sample size for each k=1,..K.
    p : array or int
        dimension/number of variables for each k=1,..,K.
    reg : str
        "GGL" for Group Graphical Lasso.
        "FGL" for Fused Graphical Lasso.
    l1 : array
        grid values for lambda1. Ideally, this is sorted in descending order.
    l2 : array, optional
        grid values for lambda2. Specify either l2 or w2.
    w2 : array, optional
        grid values for w2. 
    method : str, optional
        method for choosing the optimal grid point, either 'eBIC' or 'AIC'. The default is 'eBIC'.
    gamma : float, optional
        Parameter for the eBIC, needs to be in [0,1]. The default is 0.3.
    G : array, optional
        bookkeeping array for groups, only needed if dimensions are non-conforming. The default is None.
    latent : boolean, optional
        whether to model latent variables or not. The default is False.
    mu_range : array, optional
        grid values for mu1. Only needed when latent=True.
    ix_mu : array, optional
        shape (K,len(l1)). Indices for each element of l1 and each instance k which mu to choose from mu_range.
        Only needed when latent=True. Is computed by K_single_grid-method.
    tol : float, positive, optional
            Tolerance for the primal residual used for the solver at each grid point. The default is 1e-7.
    rtol : float, positive, optional
            Tolerance for the dual residual used for the solver at each grid point. The default is 1e-7.
    verbose : boolean, optional
        verbosity. The default is False.

    Returns
    -------
    stats : dict
        statistics of the grid search, for example BIC values, sparsity, rank of latent compinent at the grid points.
    ix : double
        index of L1/L2 grid which is selected.
    curr_best : dict
        solution of Multiple Graphical Lasso problem at the best grid point.

    """
    
    assert method in ['AIC', 'eBIC']
    assert reg in ['FGL', 'GGL']
    
    if latent:
        assert np.all(mu_range > 0)
    

    L1, L2, W2 = lambda_grid(l1, l2, w2)
    
    if verbose:
        print(L1)
        print(L2)
    
    grid1 = L1.shape[0]; grid2 = L2.shape[1]
    AIC = np.zeros((grid1, grid2))
    AIC[:] = np.nan
    
    gammas = [0.1, 0.3, 0.5, 0.7]
    gammas.append(gamma)
    gammas = list(set(gammas))
     
    BIC = dict()
    for g in gammas:
        BIC[g] = np.zeros((grid1, grid2))
        BIC[g][:] = np.nan
        
    SP = np.zeros((grid1, grid2))
    SP[:] = np.nan
    #SKIP = np.zeros((grid1, grid2), dtype = bool)
    
    
    kwargs = {'reg': reg, 'S': S, 'tol': tol, 'rtol': rtol, 'verbose': False, 'measure': False}
    if type(S) == dict:
        K = len(S.keys())
        Omega_0 = id_dict(p)
        kwargs['G'] = G
    elif type(S) == np.ndarray:
        K = S.shape[0]
        Omega_0 = id_array(K,p)
        
    kwargs['Omega_0'] = Omega_0.copy()
    
    RANK = np.zeros((K, grid1, grid2))
    
    curr_min = np.inf
    curr_best = None
    # run down the columns --> hence move g1 fastest
    # g2 indices the values in l1
    for g2 in np.arange(grid2):
        for g1 in np.arange(grid1):
      
            if verbose:
                print("Current grid point: ", (L1[g1,g2],L2[g1,g2]) )
            
            
            # set lambda1 and lambda2
            kwargs['lambda1'] = L1[g1,g2]  
            kwargs['lambda2'] = L2[g1,g2]
            
            # set the respective mu value
            if latent:
                this_mu = mu_range[ix_mu[:,g2]]
                kwargs['latent'] = True
                kwargs['mu1'] = this_mu.copy()
                if verbose:
                    print("MU values", kwargs['mu1'])
                
            # solve
            sol, info = solver(**kwargs)
            Omega_sol = sol['Omega']
            Theta_sol = sol['Theta']
            
            if latent:
                RANK[:,g1,g2] = [np.linalg.matrix_rank(sol['L'][k]) for k in np.arange(K)]
                

            
            # warm start
            kwargs['Omega_0'] = Omega_sol.copy()
            
                
            # store diagnostics
            AIC[g1,g2] = aic(S, Theta_sol, N)
            for g in gammas:
                BIC[g][g1,g2] = ebic(S, Theta_sol, N, gamma = g)
                
            SP[g1,g2] = mean_sparsity(Theta_sol)
                
            if verbose:
                print("Current eBIC grid:")
                print(BIC[gamma])
                print("Current Sparsity grid:")
                print(SP)
            
            # new best point found
            if BIC[gamma][g1,g2] < curr_min:
                print("----------New optimum found in the grid----------")
                curr_min = BIC[gamma][g1,g2]
                curr_best = sol.copy()
    
    # get optimal lambda
    if method == 'AIC':
        AIC[AIC==-np.inf] = np.nan
        ix= np.unravel_index(np.nanargmin(AIC), AIC.shape)
    
    elif method == 'eBIC':    
        for g in gammas:
            BIC[g][BIC[g]==-np.inf] = np.nan
            
        ix= np.unravel_index(np.nanargmin(BIC[gamma]), BIC[gamma].shape)
        
    stats = {'BIC': BIC, 'AIC': AIC, 'SP': SP, 'RANK': RANK, 'L1': L1, 'L2': L2, \
             'BEST': {'lambda1': L1[ix], 'lambda2': L2[ix]}, 'GAMMA': gammas}
    
    return stats, ix, curr_best

def K_single_grid(S, lambda_range, N, method = 'eBIC', gamma = 0.3, latent = False, mu_range = None, use_block = True, tol = 1e-7, rtol = 1e-7):
    """
    method for doing model selection for K single Graphical Lasso problems, using grid search and AIC/eBIC
    parameters to select: lambda1 (sparsity), mu1 (lowrank, if latent=True)
    
    A grid search on lambda1/mu1 is run on each instance independently.
    It returns two estimates:
        1) est_indv: choosing optimal lambda1/mu1 pair for each k=1,..,K independently
        2) est_uniform: choosing optimal lambda1 for all k=1,..,K uniformly and the respective optimal mu1 for each k=1,..,K independently

    Parameters
    ----------
    S : array of shape (K,p,p) or dict
        empirical covariance matrices.
    lambda_range : array
        grid values for lambda1. Ideally, this is sorted in descending order.
    N : array
        sample size for each k=1,..K.
    method : str, optional
        method for choosing the optimal grid point, either 'eBIC' or 'AIC'. The default is 'eBIC'.
    gamma : float, optional
        Parameter for the eBIC, needs to be in [0,1]. The default is 0.3.
    latent : boolean, optional
        whether to model latent variables or not. The default is False.
    mu_range : array, optional
        grid values for mu1. Only needed when latent=True.
    use_block : boolean, optional
        whether to use ADMM on each connected component. Typically, for large and sparse graphs, this is a speedup. Only possible for latent=False.
    tol : float, positive, optional
            Tolerance for the primal residual used for the solver at each grid point. The default is 1e-7.
    rtol : float, positive, optional
            Tolerance for the dual residual used for the solver at each grid point. The default is 1e-7.
    

    Returns
    -------
    est_uniform : dict
        uniformly chosen best grid point (see above for details)
    est_indv : dict
        individually chosen best grid point
    statistics : dict
        statistics of the grid search, for example BIC values, sparsity, rank of latent compinent at the grid points.

    """
    assert method in ['AIC', 'eBIC']
    
    if type(S) == dict:
        K = len(S.keys())
    elif type(S) == np.ndarray:
        K = S.shape[0]
        
    if type(N) == int:
        N = N * np.ones(K)
        
    if latent:
        assert mu_range is not None
        M = len(mu_range)
    else:
        mu_range = np.array([0])
        M = 1
    
    L = len(lambda_range)
    
    # create grid for stats, if latent = False MU is array of zeros
    MU, LAMB = np.meshgrid(mu_range, lambda_range)
    
    gammas = [0.1, 0.3, 0.5, 0.7]
    gammas.append(gamma)
    gammas = list(set(gammas))
    
    BIC = dict()
    for g in gammas:
        BIC[g] = np.zeros((K, L, M))
        BIC[g][:] = np.nan
    
    AIC = np.zeros((K, L, M))
    AIC[:] = np.nan
    
    SP = np.zeros((K, L, M))
    SP[:] = np.nan
    
    RANK = np.zeros((K,L,M))
    
    estimates = dict()
    lowrank = dict()
     
    for k in np.arange(K):
        print(f"------------Range search for instance {k}------------")
        
        if type(S) == dict:
            S_k = S[k].copy()    
        elif type(S) == np.ndarray:
            S_k = S[k,:,:].copy()
        
        best, est_k, lr_k, stats_k = single_grid_search(S = S_k, lambda_range = lambda_range, N = N[k], method = method, gamma = gamma, \
                                                             latent = latent, mu_range = mu_range, use_block = use_block, tol = tol, rtol = rtol)
        estimates[k] = est_k.copy()
        lowrank[k] = lr_k.copy()
        
        for g in gammas:
            BIC[g][k,:,:] = stats_k['BIC'][g].copy()
        AIC[k,:,:] = stats_k['AIC'].copy()
        SP[k,:,:] = stats_k['SP'].copy()
        RANK[k,:,:] = stats_k['RANK'].copy()
        
                
    # get optimal low rank for each lambda
    tmpBIC = dict()
    for g in gammas:
        tmpBIC[g] = np.zeros((K,L))
        tmpBIC[g][:] = np.nan
    
    tmpAIC = np.zeros((K, L))
    tmpAIC[:] = np.nan
    
    ix_mu = np.zeros((K,L), dtype = int)
    
    # for each lambda, get optimal mu
    for k in np.arange(K):
        for j in np.arange(L):
            
            if method == 'AIC':
                ix_mu[k,j] = np.nanargmin(AIC[k,j,:])
                
            elif method == 'eBIC':
                ix_mu[k,j] = np.nanargmin(BIC[gamma][k,j,:])
                
            tmpAIC[k,j] = AIC[k,j,ix_mu[k,j]]
            
            for g in gammas:
                tmpBIC[g][k,j] = BIC[g][k,j,ix_mu[k,j]]
                
    # get optimal lambda (uniform over k =1,..,K and individual)
    if method == 'AIC':
        tmpAIC[tmpAIC==-np.inf] = np.nan
        ix_uniform = np.nanargmin(tmpAIC.sum(axis=0))
        ix_indv = np.nanargmin(tmpAIC, axis = 1)
        
    elif method == 'eBIC':    
        for g in gammas:
            tmpBIC[g][tmpBIC[g]==-np.inf] = np.nan
        
        ix_uniform = np.nanargmin(tmpBIC[gamma].sum(axis=0))
        ix_indv = np.nanargmin(tmpBIC[gamma], axis = 1)
        
    # create the two estimators
    est_uniform = dict()
    est_indv = dict()
    est_uniform['Theta'] = dict()
    est_indv['Theta'] = dict()
    if latent:
        est_uniform['L'] = dict()
        est_indv['L'] = dict()
        
    for k in np.arange(K):
        
        est_uniform['Theta'][k] = estimates[k][ix_uniform, ix_mu[k,ix_uniform] , :,:]
        est_indv['Theta'][k] = estimates[k][ix_indv[k], ix_mu[k,ix_indv[k]], :, :]
        
        if latent:
            est_uniform['L'][k] = lowrank[k][ix_uniform, ix_mu[k,ix_uniform] , :,:]
            est_indv['L'][k] = lowrank[k][ix_indv[k], ix_mu[k,ix_indv[k]], :, :]
            
    
    if type(S) == np.ndarray:
        est_indv['Theta'] = np.stack([e for e in est_indv['Theta'].values()])
        est_uniform['Theta'] = np.stack([e for e in est_uniform['Theta'].values()])
        if latent:
            est_indv['L'] = np.stack([e for e in est_indv['L'].values()])
            est_uniform['L'] = np.stack([e for e in est_uniform['L'].values()])
    
    statistics = {'BIC': BIC[gamma], 'AIC': AIC, 'SP': SP, 'RANK': RANK, \
                  'LAMB': LAMB, 'MU': MU,\
                  'ix_uniform': ix_uniform, 'ix_indv': ix_indv, 'ix_mu': ix_mu}
    
    return est_uniform, est_indv, statistics


def single_grid_search(S, lambda_range, N, method = 'eBIC', gamma = 0.3, latent = False, mu_range = None, use_block = True, tol = 1e-7, rtol = 1e-7):
    """
    method for model selection for SGL problem, doing grid search and selection via eBIC or AIC

    Parameters
    ----------
    S : array of shape (p,p)
        empirical covariance matrix.
    lambda_range : array
        range of lambda1 values (sparsity regularization parameter). Ideally, this is sorted in descending order.
    N : int
        sample size.
    method : str, optional
        method for choosing the optimal grid point, either 'eBIC' or 'AIC'. The default is 'eBIC'.
    gamma : float, optional
        Parameter for the eBIC, needs to be in [0,1]. The default is 0.3.
    latent : boolean, optional
        whether to model latent variables or not. The default is False.  
    mu_range : array, optional
        range of mu1 values (low rank regularization parameter). Only needed when latent = True.
    use_block : boolean, optional
        whether to use ADMM on each connected component. Typically, for large and sparse graphs, this is a speedup. Only possible for latent=False.
    tol : float, positive, optional
            Tolerance for the primal residual used for the solver at each grid point. The default is 1e-7.
    rtol : float, positive, optional
            Tolerance for the dual residual used for the solver at each grid point. The default is 1e-7.
    
    
    Returns
    -------
    best_sol : dict
        solution of SGL problem at best grid point.
    estimates : array
        solutions of Theta variable at all grid points.
    lowrank : array
        solutions of L variable at all grid points.
    stats : dict
        statistics of the grid search, for example BIC values, sparsity, rank of latent compinent at the grid points.

    """
    p = S.shape[0]
    
    if latent:
        assert mu_range is not None
        M = len(mu_range)
    else:
        mu_range = np.array([0])
        M = 1
       
    L = len(lambda_range)
    
    gammas = [0.1, 0.3, 0.5, 0.7]
    gammas.append(gamma)
    gammas = list(set(gammas))
    
    # create grid for stats, if latent = False MU is array of zeros
    MU, LAMB = np.meshgrid(mu_range, lambda_range)
        
    BIC = dict()
    for g in gammas:
        BIC[g] = np.zeros((L, M))
        BIC[g][:] = np.nan
    
    AIC = np.zeros((L, M))
    AIC[:] = np.inf
    
    SP = np.zeros((L, M))
    SP[:] = np.inf
    
    RANK = np.zeros((L,M))
    
    kwargs = {'S': S, 'Omega_0': np.eye(p), 'X_0': np.eye(p), 'tol': tol, 'rtol': rtol,\
              'verbose': False, 'measure': False}
    
    estimates = np.zeros((L,M,p,p))
    lowrank = np.zeros((L,M,p,p))
    
    # start range search
    for j in np.arange(L):
        kwargs['lambda1'] = lambda_range[j]
        
        for m in np.arange(M):
            if latent:
                kwargs['mu1'] = mu_range[m]
                kwargs['latent'] = True
            
            if use_block and not latent:
                sol = block_SGL(**kwargs)
            else:
                sol, _ = ADMM_SGL(**kwargs)
            
            Theta_sol = sol['Theta']
            estimates[j,m,:,:] = Theta_sol.copy()
            
            if latent:
                lowrank[j,m,:,:] = sol['L'].copy()
                RANK[j,m] = np.linalg.matrix_rank(sol['L'])
            
            # warm start
            kwargs['Omega_0'] = sol['Omega'].copy()
            # as X is scaled with rho, warm starting the dual variables can go wrong when rho is adapted
            
            
            AIC[j,m] = aic_single(S, Theta_sol, N)
            for g in gammas:
                BIC[g][j, m] = ebic_single(S, Theta_sol, N, gamma = g)
                
            SP[j,m] = sparsity(Theta_sol)
            

    AIC[AIC==-np.inf] = np.nan
    for g in gammas:
        BIC[g][BIC[g]==-np.inf] = np.nan
    
    if method == 'AIC':    
        ix= np.unravel_index(np.nanargmin(AIC), AIC.shape)
    elif method == 'eBIC':        
        ix= np.unravel_index(np.nanargmin(BIC[gamma]), BIC[gamma].shape)
        
        
    best_sol = dict()
    best_sol['Theta'] = estimates[ix]
    if latent:
        best_sol['L'] = lowrank[ix]
    

    stats = {'BIC': BIC, 'AIC': AIC, 'SP': SP, 'RANK': RANK, 'LAMBDA': LAMB, 'MU': MU, \
             'BEST': {'lambda1':LAMB[ix], 'mu1': MU[ix]}, 'GAMMA': gammas}
            
    return best_sol, estimates, lowrank, stats

################################################################
## CRITERIA AIC/EBIC
################################################################
    
def aic(S, Theta, N):
    """
    AIC information criterion after Danaher et al.
    excludes the diagonal
    """
    if type(S) == dict:
        aic = aic_dict(S, Theta, N)
    elif type(S) == np.ndarray:
        aic = aic_array(S, Theta, N)
    else:
        raise KeyError("Not a valid input type -- should be either dictionary or ndarray")
    
    return aic

def aic_dict(S, Theta, N):
    """
    S, Theta are dictionaries
    N is array of sample sizes
    """
    K = len(S.keys())
     
    aic = 0
    for k in np.arange(K):
        aic += aic_single(S[k], Theta[k], N[k])
    return aic

def aic_array(S,Theta, N):
    (K,p,p) = S.shape
    
    if type(N) == int:
        N = np.ones(K) * N
    
    aic = 0
    for k in np.arange(K):
        aic += aic_single(S[k,:,:], Theta[k,:,:], N[k])

    return aic

def aic_single(S, Theta, N):
    (p,p) = S.shape
        
    A = adjacency_matrix(Theta , t = 1e-5)
    aic = N*Sdot(S, Theta) - N*robust_logdet(Theta) + A.sum()
    
    return aic

################################################################
    
def ebic(S, Theta, N, gamma = 0.5):
    """
    extended BIC after Drton et al.
    """
    if type(S) == dict:
        ebic = ebic_dict(S, Theta, N, gamma)
    elif type(S) == np.ndarray:
        ebic = ebic_array(S, Theta, N, gamma)
    else:
        raise KeyError("Not a valid input type -- should be either dictionary or ndarray")
    
    return ebic

def ebic_single(S,Theta, N, gamma):
    (p,p) = S.shape
        
    A = adjacency_matrix(Theta , t = 1e-5)
    bic = N*Sdot(S, Theta) - N*robust_logdet(Theta) + A.sum()/2 * (np.log(N)+ 4*np.log(p)*gamma)
    
    return bic

def ebic_array(S, Theta, N, gamma):
    (K,p,p) = S.shape
    
    if type(N) == int:
        N = np.ones(K) * N
        
    bic = 0
    for k in np.arange(K):
        bic += ebic_single(S[k,:,:], Theta[k,:,:], N[k], gamma)
    return bic

def ebic_dict(S, Theta, N, gamma):
    """
    S, Theta are dictionaries
    N is array of sample sizes
    """
    K = len(S.keys())
   
    bic = 0
    for k in np.arange(K):
        bic += ebic_single(S[k], Theta[k], N[k], gamma)
        
    return bic
        
def robust_logdet(A, t = 1e-6):
    """
    slogdet returns always a finite number if the lowest EV is not EXACTLY 0
    because of numerical inaccuracies we want to avoid that behaviour but also avoid overflows
    """
    D,Q = np.linalg.eigh(A)
    if D.min() <= t:
        print("WARNING: solution may not be positive definite")
        return -np.inf
    else:
        l = np.linalg.slogdet(A)
        return l[0]*l[1]
    
