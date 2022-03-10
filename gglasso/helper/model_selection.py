"""
author: Fabian Schaipp
"""

import numpy as np

from .basic_linalg import Sdot
from .utils import mean_sparsity, sparsity

from .utils import get_K_identity as id_array
from .ext_admm_helper import get_K_identity as id_dict
from ..solver.single_admm_solver import ADMM_SGL, block_SGL


# default parameters
DEFAULT_GAMMAS = [0.1, 0.3, 0.5, 0.7]
TAU_MIN = 1e-12
N_TAU = 20
      
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
                G = None, latent = False, mu_range = None, ix_mu = None, thresholding = False, tol = 1e-7, rtol = 1e-7, verbose = False):
    """
    method for doing model selection for MGL problems using grid search and AIC/eBIC
    parameters to select: lambda1 (sparsity), lambda2 (group sparsity or total variation)
    
    In the grid lambda1 changes over columns, lambda2 over the rows.
    The grid is ran columnwise, i.e. hold l1 constant and change l2.
    
    
    Parameters
    ----------
    solver : solver method 
        ``ADMM_MGL`` or ``ext_ADMM_MGL``.
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
    thresholding : boolean, optional
        whether to tune a thresholded estimator for each (lambda1,lambda2) pair. See https://arxiv.org/pdf/2104.06389v1.pdf for details.
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
    
    if type(S) == dict:
        K = len(S.keys())
    elif type(S) == np.ndarray:
        K = S.shape[0]
        
    assert len(N) == K, f"N must be given as array, is given as {N}."
    
    if latent:
        assert np.all(mu_range > 0)
      
    L1, L2, W2 = lambda_grid(l1, l2, w2)
    
    if verbose:
        print("Grid of lambda1/lambda2:")
        print(L1)
        print(L2)
    
    grid1 = L1.shape[0]; grid2 = L2.shape[1]
    AIC = np.nan*np.zeros((grid1, grid2))
    
    # use default gammas plus the one for eBIC selection
    gammas = DEFAULT_GAMMAS
    gammas.append(gamma)
    gammas = list(set(gammas))
     
    BIC = dict()
    for g in gammas:
        BIC[g] = np.nan*np.zeros((grid1, grid2))
        
    SP = np.nan*np.zeros((grid1, grid2))
    #SKIP = np.zeros((grid1, grid2), dtype = bool)
    RANK = np.nan*np.zeros((K, grid1, grid2))
    
    if thresholding:
        TAU = np.zeros((K, grid1, grid2))
    else:
        TAU = None
    
    # solver kwargs
    kwargs = {'reg': reg, 'S': S, 'tol': tol, 'rtol': rtol, 'verbose': False, 'measure': False}
    if type(S) == dict:
        Omega_0 = id_dict(p)
        kwargs['G'] = G
    elif type(S) == np.ndarray:
        Omega_0 = id_array(K,p)
        
    kwargs['Omega_0'] = Omega_0.copy()
    
    
    curr_min = np.inf
    curr_best = None
    
    if thresholding:
        _no_thr_curr_min = np.inf
        _no_thr_best_params = None
        _no_thr_curr_best = None # store also the best solution before thresholding
            
    #==================================================
    # MAIN LOOP
    #
    # run down the columns --> move g1 fastest
    # g2 indices the values in l1
    for g2 in np.arange(grid2):
        for g1 in np.arange(grid1):
      
            # set lambda1 and lambda2
            kwargs['lambda1'] = L1[g1,g2]  
            kwargs['lambda2'] = L2[g1,g2]
            
            # set the respective mu value
            if latent:
                this_mu = mu_range[ix_mu[:,g2]]
                kwargs['latent'] = True
                kwargs['mu1'] = this_mu.copy()
                #print("MU values", kwargs['mu1'])
                
            # solve
            sol, info = solver(**kwargs)
            # warm start
            kwargs['Omega_0'] = sol['Omega'].copy()
            
            # thresholding
            if thresholding:
                # store also the best solution without solution
                _no_thr_this_score = ebic(S, sol['Theta'], N, gamma = gamma)
                if  _no_thr_this_score < _no_thr_curr_min:
                    _no_thr_best_params = {'lambda1': L1[g1,g2], 'lambda2':L2[g1,g2]}
                    _no_thr_curr_min = _no_thr_this_score
                    _no_thr_curr_best = sol.copy()
                
                # now tune threshold
                sol['Theta'], opt_tau, _ = tune_multiple_threshold(sol['Theta'], S, N, tau_range = None,\
                                                                   method = method, gamma = gamma)
                TAU[:,g1,g2] = opt_tau
            
            #########################################         
            # store diagnostics
            AIC[g1,g2] = aic(S, sol['Theta'], N)
            for g in gammas:
                BIC[g][g1,g2] = ebic(S, sol['Theta'], N, gamma = g)
            
            if latent:
                RANK[:,g1,g2] = [np.linalg.matrix_rank(sol['L'][k]) for k in np.arange(K)]
            
            SP[g1,g2] = mean_sparsity(sol['Theta'])
                        
            # new best point found
            if method == 'eBIC':
                if BIC[gamma][g1,g2] < curr_min:
                    curr_min = BIC[gamma][g1,g2]
                    curr_best = sol.copy()
            elif method == 'AIC':
                 if AIC[g1,g2] < curr_min:
                    curr_min = AIC[g1,g2]
                    curr_best = sol.copy()
        
            if verbose:
                print(f"Grid point: (l1,l2): {(L1[g1,g2],L2[g1,g2])}, sparsity: {np.round(SP[g1,g2],3)}, best score: {np.round(curr_min,1)}")
            
    # get optimal lambda
    if method == 'AIC':
        AIC[AIC==-np.inf] = np.nan
        ix= np.unravel_index(np.nanargmin(AIC), AIC.shape)   
    elif method == 'eBIC':    
        for g in gammas:
            BIC[g][BIC[g]==-np.inf] = np.nan            
        ix= np.unravel_index(np.nanargmin(BIC[gamma]), BIC[gamma].shape)
    
    if verbose:
        print(f"Best regularization parameters: (l1,l2): {(L1[ix],L2[ix])}")
        
    stats = {'BIC': BIC, 'AIC': AIC, 'SP': SP, 'RANK': RANK, 'TAU': TAU, 'L1': L1, 'L2': L2, \
             'BEST': {'lambda1': L1[ix], 'lambda2': L2[ix]}, 'GAMMA': gammas}
    
    if thresholding:
        stats['NO_THRESHOLDING_SOL'] = _no_thr_curr_best
        stats['NO_THRESHOLDING_BEST'] = _no_thr_best_params
    
    return stats, ix, curr_best

def K_single_grid(S, lambda_range, N, method = 'eBIC', gamma = 0.3, latent = False, mu_range = None,\
                  thresholding = False, use_block = True, store_all = True, tol = 1e-7, rtol = 1e-7):
    """
    method for doing model selection for K single Graphical Lasso problems, using grid search and AIC/eBIC
    parameters to select: lambda1 (sparsity), mu1 (lowrank, if latent=True)
    
    A grid search on lambda1/mu1 is run on each instance independently.
    It returns 
        1) est_indv: choosing optimal lambda1/mu1 pair for each k=1,..,K independently
        2) est_uniform: Only if ``store_all = True``. Choosing optimal lambda1 for all k=1,..,K uniformly and the respective optimal mu1 for each k=1,..,K independently.
                        Caution as you might run into memory issues.

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
    thresholding : boolean, optional
        whether to tune a thresholded estimator for each (lambda1,mu1) pair. See https://arxiv.org/pdf/2104.06389v1.pdf for details.
    use_block : boolean, optional
        whether to use ADMM on each connected component. Typically, for large and sparse graphs, this is a speedup. Only possible for latent=False.
    store_all : boolean, optional
        If you want to compute est_uniform, set to True. When only best mu for each k=1,..,K and lambda1 is needed, can be set to False. The default is True.
    tol : float, positive, optional
        Tolerance for the primal residual used for the solver at each grid point. The default is 1e-7.
    rtol : float, positive, optional
        Tolerance for the dual residual used for the solver at each grid point. The default is 1e-7.
    

    Returns
    -------
    est_uniform : dict (or None)
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
        
    assert len(N) == K, f"N must be given as array, is given as {N}."
        
    if latent:
        assert mu_range is not None
        _M = len(mu_range)
    else:
        mu_range = np.array([0])
        _M = 1
    
    _L = len(lambda_range)
    
    # create grid for stats, if latent = False MU is array of zeros
    MU, LAMB = np.meshgrid(mu_range, lambda_range)
    
    gammas = DEFAULT_GAMMAS
    gammas.append(gamma)
    gammas = list(set(gammas))
    
    BIC = dict()
    for g in gammas:
        BIC[g] = np.nan*np.zeros((K,_L,_M))
        
    AIC = np.nan*np.zeros((K,_L,_M))
    SP = np.nan*np.zeros((K,_L,_M))
    RANK = np.zeros((K,_L,_M))
    
    estimates = dict()
    lowrank = dict()
    est_indv = dict()
    est_indv['Theta'] = dict()
    if latent: 
        est_indv['L'] = dict()
    
    ###########################################
    # MAIN LOOP
    ###########################################
    for k in np.arange(K):
        print(f"------------Range search for instance {k}------------")
        
        if type(S) == dict:
            S_k = S[k].copy()    
        elif type(S) == np.ndarray:
            S_k = S[k,:,:].copy()
        
        best, est_k, lr_k, stats_k = single_grid_search(S = S_k, lambda_range = lambda_range, N = N[k], method = method, gamma = gamma, \
                                                        latent = latent, mu_range = mu_range, thresholding = thresholding,\
                                                        use_block = use_block, store_all = store_all, tol = tol, rtol = rtol)
        
        #store best individual estimator
        est_indv['Theta'][k] = best['Theta'].copy() 
        if latent:
            est_indv['L'][k] = best['L'].copy()
        
        if store_all:
            estimates[k] = est_k.copy()
            lowrank[k] = lr_k.copy()
        
        for g in gammas:
            BIC[g][k,:,:] = stats_k['BIC'][g].copy()
        AIC[k,:,:] = stats_k['AIC'].copy()
        SP[k,:,:] = stats_k['SP'].copy()
        RANK[k,:,:] = stats_k['RANK'].copy()
    
    ###########################################
    # get optimal low rank for each lambda
    tmpSCORE = np.zeros((K,_L))
    tmpSCORE[:] = np.nan

    ix_mu = np.zeros((K,_L), dtype = int)
    # for each lambda, get optimal mu
    for k in np.arange(K):
        for j in np.arange(_L):       
            if method == 'AIC':
                ix_mu[k,j] = np.nanargmin(AIC[k,j,:])     
                tmpSCORE[k,j] = AIC[k,j,ix_mu[k,j]]
            elif method == 'eBIC':
                ix_mu[k,j] = np.nanargmin(BIC[gamma][k,j,:])
                tmpSCORE[k,j] = BIC[gamma][k,j,ix_mu[k,j]]
    
    # get optimal lambda (uniform over k=1,..,K and individual)
    tmpSCORE[tmpSCORE==-np.inf] = np.nan
    ix_uniform = np.nanargmin(tmpSCORE.sum(axis=0))
    ix_indv = np.nanargmin(tmpSCORE, axis = 1)
         
    # stack in case of array
    if type(S) == np.ndarray:
        est_indv['Theta'] = np.stack([e for e in est_indv['Theta'].values()])
        if latent:
            est_indv['L'] = np.stack([e for e in est_indv['L'].values()])
    
    ###########################################
    # create est_uniform
    if store_all:          
        est_uniform = dict()
        est_uniform['Theta'] = dict()
        if latent: 
            est_uniform['L'] = dict()
        
        for k in np.arange(K):  
            est_uniform['Theta'][k] = estimates[k][ix_uniform, ix_mu[k,ix_uniform] , :,:]           
            if latent:
                est_uniform['L'][k] = lowrank[k][ix_uniform, ix_mu[k,ix_uniform] , :,:]
                   
        if type(S) == np.ndarray:
            est_uniform['Theta'] = np.stack([e for e in est_uniform['Theta'].values()])
            if latent:
                est_uniform['L'] = np.stack([e for e in est_uniform['L'].values()])
    else:
        est_uniform = None
        
    statistics = {'BIC': BIC[gamma], 'AIC': AIC, 'SP': SP, 'RANK': RANK, \
                  'LAMB': LAMB, 'MU': MU,\
                  'ix_uniform': ix_uniform, 'ix_indv': ix_indv, 'ix_mu': ix_mu}
    
            
    return est_uniform, est_indv, statistics


def single_grid_search(S, lambda_range, N, method = 'eBIC', gamma = 0.3, latent = False, mu_range = None,\
                       thresholding = False, use_block = True, store_all = True, tol = 1e-7, rtol = 1e-7):
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
    thresholding : boolean, optional
        whether to tune a thresholded estimator for each (lambda1,mu1) pair. See https://arxiv.org/pdf/2104.06389v1.pdf for details.
    use_block : boolean, optional
        whether to use ADMM on each connected component. Typically, for large and sparse graphs, this is a speedup. Only possible for latent=False.
    store_all : boolean, optional
        whether the solution at any grid point is stored. This might be needed if a comparative estimator shall be computed. The default is False.
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
        _M = len(mu_range)
    else:
        mu_range = np.array([0])
        _M = 1
       
    _L = len(lambda_range)
    
    gammas = DEFAULT_GAMMAS
    gammas.append(gamma)
    gammas = list(set(gammas))
    
    # create grid for stats, if latent = False MU is array of zeros
    MU, LAMB = np.meshgrid(mu_range, lambda_range)
        
    BIC = dict()
    for g in gammas:
        BIC[g] = np.nan*np.zeros((_L, _M))
        
    AIC = np.nan*np.zeros((_L, _M))
    SP = np.nan*np.zeros((_L, _M))
    RANK = np.zeros((_L,_M))
    
    if thresholding:
        TAU = np.zeros((_L, _M))
    else:
        TAU = None
    
    kwargs = {'S': S, 'Omega_0': np.eye(p), 'X_0': np.eye(p), 'tol': tol, 'rtol': rtol,\
              'verbose': False, 'measure': False}
    
    if store_all:
        estimates = np.zeros((_L,_M,p,p))
        lowrank = np.zeros((_L,_M,p,p))
    else:
        estimates = None
        lowrank = None
    
    best_sol = dict()
    curr_min = np.inf
    
    # start range search
    for j in np.arange(_L):
        kwargs['lambda1'] = lambda_range[j]
        
        for m in np.arange(_M):
            if latent:
                kwargs['mu1'] = mu_range[m]
                kwargs['latent'] = True
            
            if use_block and not latent:
                sol = block_SGL(**kwargs)
            else:
                sol, _ = ADMM_SGL(**kwargs)
                        
            kwargs['Omega_0'] = sol['Omega'].copy() # warm start
            # as X is scaled with rho, warm starting the dual variables can go wrong when rho is adapted
            
            if latent:
                if store_all:
                    lowrank[j,m,:,:] = sol['L'].copy()
                RANK[j,m] = np.linalg.matrix_rank(sol['L'])
            
            # tune optimal threshold, changes sol['Theta']
            if thresholding:
                sol['Theta'], opt_tau, _ = tune_threshold(sol['Theta'], S, N,\
                                                          tau_range = None, method = method, gamma = gamma)
                TAU[j,m] = opt_tau
             
            AIC[j,m] = aic_single(S, sol['Theta'], N)
            for g in gammas:
                BIC[g][j, m] = ebic_single(S, sol['Theta'], N, gamma = g)
            
            SP[j,m] = sparsity(sol['Theta'])
            
            if store_all:
                estimates[j,m,:,:] = sol['Theta'].copy()
            
            # new best point found
            if method == 'eBIC':
                if BIC[gamma][j,m] < curr_min:
                    curr_min = BIC[gamma][j,m]
                    best_sol = sol.copy()
            elif method == 'AIC':
                 if AIC[j,m] < curr_min:
                    curr_min = AIC[j,m]
                    best_sol = sol.copy()
        

    AIC[AIC==-np.inf] = np.nan
    for g in gammas:
        BIC[g][BIC[g]==-np.inf] = np.nan
    
    if method == 'AIC':    
        ix = np.unravel_index(np.nanargmin(AIC), AIC.shape)
    elif method == 'eBIC':        
        ix = np.unravel_index(np.nanargmin(BIC[gamma]), BIC[gamma].shape)
        
    
    stats = {'BIC': BIC, 'AIC': AIC, 'SP': SP, 'RANK': RANK, 'LAMBDA': LAMB, 'MU': MU, 'TAU': TAU,\
             'BEST': {'lambda1': LAMB[ix], 'mu1': MU[ix]}, 'GAMMA': gammas}
            
    return best_sol, estimates, lowrank, stats

#######################################
## THRESHOLDING
#######################################

def thresholding(A, tau):
    """
    thresholding array A by tau
    """
    mask = (np.abs(A) > tau)
    np.fill_diagonal(mask,1.) # dont threshold on diagonal
       
    return A*mask

def tune_threshold(Theta, S, N, tau_range = None, method = 'eBIC', gamma = 0.1):
    """
    Pick the best threshold for 2d-array according to eBIC or AIC.
    """
    if tau_range is None:
        # diagonal is upper bound as this would make Theta indefinite.                   
        #tau_range = np.linspace(TAU_MIN, np.diag(Theta).min()*0.9, N_TAU) 
        tau_range = np.logspace(-12,-1,N_TAU)
        
    assert np.all(tau_range > 0)

    scores = np.zeros(len(tau_range))
    
    for j in range(len(tau_range)):
        tau = tau_range[j]
        if method == 'eBIC':
            E = ebic(S, thresholding(Theta, tau), N, gamma = gamma)
        elif method == 'AIC':
            E = aic(S, thresholding(Theta, tau), N)
        scores[j] = E
        
    scores[scores==np.inf] = np.nan
    
    opt_ix = np.nanargmin(scores)
    opt_tau = tau_range[opt_ix]
    
    t_Theta = thresholding(Theta, opt_tau)
    return t_Theta, opt_tau, scores

def tune_multiple_threshold(Theta, S, N, tau_range, method = 'eBIC', gamma = 0.1):
    """
    Pick the best threshold for 3d-array or dict according to eBIC or AIC. 
    """
    if type(S) == dict:
        K = len(S.keys())
    elif type(S) == np.ndarray:
        K = S.shape[0]
    
    t_Theta = Theta.copy()
    score = dict()
    tau = np.zeros(K)
    
    for k in np.arange(K):
        Th_k, tau_k, scores_k = tune_threshold(Theta[k], S[k], N[k], tau_range, method, gamma)
        score[k] = scores_k
        tau[k] = tau_k
        t_Theta[k] = Th_k
    
    return t_Theta, tau, score

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
        if len(S.shape) == 3:
            aic = aic_array(S, Theta, N)
        else:
            aic = aic_single(S, Theta, N)          
    else:
        raise KeyError("Not a valid input type -- should be either dictionary or ndarray")
    
    return aic

def aic_dict(S, Theta, N):
    """
    S, Theta are dictionaries
    N is array of sample sizes
    """
    K = len(S.keys())
    
    if isinstance(N, (int,float,np.integer,np.float)):
        N = np.ones(K) * N
        
    aic = 0
    for k in np.arange(K):
        aic += aic_single(S[k], Theta[k], N[k])
    return aic

def aic_array(S,Theta, N):
    (K,p,p) = S.shape
    
    if isinstance(N, (int,float,np.integer,np.float)):
        N = np.ones(K) * N
    
    aic = 0
    for k in np.arange(K):
        aic += aic_single(S[k,:,:], Theta[k,:,:], N[k])

    return aic

def aic_single(S, Theta, N):
    (p,p) = S.shape
    assert isinstance(N, (int,float,np.integer,np.float))
        
    # count upper diagonal non-zero entries
    E = (np.count_nonzero(Theta) - p)/2
    aic = N*Sdot(S, Theta) - N*robust_logdet(Theta) + E
    
    return aic

################################################################
    
def ebic(S, Theta, N, gamma = 0.5):
    """
    extended BIC after Drton et al.
    """
    if type(S) == dict:
        ebic = ebic_dict(S, Theta, N, gamma)
    elif type(S) == np.ndarray:
        if len(S.shape) == 3:
            ebic = ebic_array(S, Theta, N, gamma)
        else:
            ebic = ebic_single(S, Theta, N, gamma)
    else:
        raise KeyError("Not a valid input type -- should be either dictionary or ndarray")
    
    return ebic

def ebic_single(S, Theta, N, gamma):
    (p,p) = S.shape
    assert isinstance(N, (int,float,np.integer,np.float))
    
    # count upper diagonal non-zero entries
    E = (np.count_nonzero(Theta) - p)/2
    bic = N*Sdot(S, Theta) - N*robust_logdet(Theta) + E*(np.log(N)+ 4*np.log(p)*gamma)
    
    return bic

def ebic_array(S, Theta, N, gamma):
    (K,p,p) = S.shape   
    if isinstance(N, (int,float,np.integer,np.float)):
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
    if isinstance(N, (int,float,np.integer,np.float)):
        N = np.ones(K) * N
    
    bic = 0
    for k in np.arange(K):
        bic += ebic_single(S[k], Theta[k], N[k], gamma)
        
    return bic
        
def robust_logdet(A, t=1e-12):
    """
    slogdet returns always a finite number if the lowest EV is not EXACTLY 0
    because of numerical inaccuracies we want to return inf if smallest eigenvalue is below threshold t
    """
    D = np.linalg.eigvalsh(A)
    if D.min() <= t:
        return -np.inf
    else:
        l = np.linalg.slogdet(A)
        return l[0]*l[1]
    
