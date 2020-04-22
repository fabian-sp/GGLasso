"""
author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gglasso.helper.basic_linalg import Sdot, adjacency_matrix
from gglasso.helper.experiment_helper import mean_sparsity, sparsity, consensus

from gglasso.helper.experiment_helper import get_K_identity as id_array
from gglasso.helper.ext_admm_helper import get_K_identity as id_dict
from gglasso.solver.single_admm_solver import ADMM_SGL


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

def grid_search(solver, S, N, p, reg, l1, method= 'eBIC', l2 = None, w2 = None, G = None, latent = False, mu = None, ix_mu = None):
    """
    method for doing model selection using grid search and AIC/eBIC
    we work the grid columnwise, i.e. hold l1 constant and change l2
    
    set latent = True if you want to include latent factors. 
    ix_mu: array of shape K, len(l1): indicates for each lambda and each instance which value in mu we use (can be obtained with single_range_search)
    mu: arrays of possible mu values
    """
    
    assert method in ['AIC', 'eBIC']
    assert reg in ['FGL', 'GGL']
    
    if latent:
        assert np.all(mu > 0)

    L1, L2, W2 = lambda_grid(l1, l2, w2)
    
    print(L1)
    print(L2)
    grid1 = L1.shape[0]; grid2 = L2.shape[1]
    AIC = np.zeros((grid1, grid2))
    AIC[:] = np.nan
    
    gammas = [0.1, 0.3, 0.5, 0.7]
    # determine the gamma you want for the returned estimate
    gamma_ix = 1
    BIC = np.zeros((len(gammas), grid1, grid2))
    BIC[:] = np.nan
    
    SP = np.zeros((grid1, grid2))
    SP[:] = np.nan
    SKIP = np.zeros((grid1, grid2), dtype = bool)
    
    
    kwargs = {'reg': reg, 'S': S, 'eps_admm': 1e-3, 'verbose': False, 'measure': False}
    if type(S) == dict:
        K = len(S.keys())
        Omega_0 = id_dict(p)
        kwargs['G'] = G
    elif type(S) == np.ndarray:
        K = S.shape[0]
        Omega_0 = id_array(K,p)
        
    kwargs['Omega_0'] = Omega_0.copy()
    
    # unique edges
    UQED = np.zeros((grid1, grid2))
    RANK = np.zeros((K, grid1, grid2))
    
    curr_min = np.inf
    curr_best = None
    # run down the columns --> hence move g1 fastest
    # g2 indices the values in l1
    for g2 in np.arange(grid2):
        for g1 in np.arange(grid1):
      
            print("Current grid point: ", (L1[g1,g2],L2[g1,g2]) )
            if SKIP[g1,g2]:
                print("SKIP")
                continue
           
            kwargs['lambda1'] = L1[g1,g2]            
            # set the respective mu value
            if latent:
                this_mu = mu[ix_mu[:,g2]]
                kwargs['latent'] = True
                kwargs['mu1'] = this_mu
                print("MU values", kwargs['mu1'])
                
                
            kwargs['lambda2'] = L2[g1,g2]
            
            sol, info = solver(**kwargs)
            Omega_sol = sol['Omega']
            Theta_sol = sol['Theta']
            
            if latent:
                RANK[:,g1,g2] = [np.linalg.matrix_rank(sol['L'][k]) for k in np.arange(K)]
                
            #if mean_sparsity(Theta_sol) >= 0.18:
            #    SKIP[g1:, g2:] = True
            
            # warm start
            kwargs['Omega_0'] = Omega_sol.copy()
            if 'X0' in sol.keys():
                kwargs['X0'] = sol['X0'].copy()
                kwargs['X1'] = sol['X1'].copy()
            elif 'X' in sol.keys():
                kwargs['X_0'] = sol['X'].copy()
                
                
            AIC[g1,g2] = aic(S, Theta_sol, N)
            for j in np.arange(len(gammas)):
                BIC[j, g1,g2] = ebic(S, Theta_sol, N, gamma = gammas[j])
                
            SP[g1,g2] = mean_sparsity(Theta_sol)
            
            #if np.all(G!= None):
            #    nnz = consensus(Theta_sol, G)
            #    UQED[g1,g2] = (nnz >= 1).sum()
                
            
            print("Current eBIC grid:")
            print(BIC[gamma_ix,:,:])
            print("Current Sparsity grid:")
            print(SP)
            
            if BIC[gamma_ix,g1,g2] < curr_min:
                print("----------New optimum found in the grid----------")
                curr_min = BIC[gamma_ix,g1,g2]
                curr_best = sol.copy()
    
    # get optimal lambda
    if method == 'AIC':
        AIC[AIC==-np.inf] = np.nan
        ix= np.unravel_index(np.nanargmin(AIC), AIC.shape)
    elif method == 'eBIC':    
        BIC[BIC==-np.inf] = np.nan
        ix= np.unravel_index(np.nanargmin(BIC[gamma_ix,:,:]), BIC[gamma_ix,:,:].shape)
        
    stats = {'BIC': BIC, 'AIC': AIC, 'SP': SP, 'RANK': RANK, 'L1': L1, 'L2': L2}
    
    return stats, ix, curr_best

def single_range_search(S, L, N, method = 'eBIC', latent = False, mu = None):
    """
    method for doing model selection for sungle Graphical Lasso estimation
    it returns two estimates, one with the individual optimal reg. param. for each instance and one with the uniform optimal
    L: range of lambda values
    N: vector with sample sizes for each instance
    
    latent: boolean which indicates if low rank term should be estimated (i.e. Latent Variable Graphical Lasso)
    mu: range of penalty parameters for trace norm (only needed if latent = True)
    """
    
    if type(S) == dict:
        K = len(S.keys())
    elif type(S) == np.ndarray:
        K = S.shape[0]
        
    if type(N) == int:
        N = N * np.ones(K)
        
    if latent:
        assert mu is not None
        M = len(mu)
    else:
        M = 1
        
    r = len(L)
    
    gammas = [0.1, 0.3, 0.5, 0.7]
    # determine the gamma you want for the returned estimate
    gamma_ix = 1
    BIC = np.zeros((len(gammas), K, r, M))
    BIC[:] = np.nan
    
    AIC = np.zeros((K, r, M))
    AIC[:] = np.nan
    
    SP = np.zeros((K, r, M))
    SP[:] = np.nan
    
    RANK = np.zeros((K,r,M))
    
    estimates = dict()
    lowrank = dict()
    
    kwargs = {'eps_admm': 1e-4, 'verbose': False, 'measure': False}
    
    for k in np.arange(K):
        print(f"------------Range search for instance {k}------------")
        
        if type(S) == dict:
            S_k = S[k].copy()
            kwargs['S'] = S_k.copy()
        elif type(S) == np.ndarray:
            S_k = S[k,:,:].copy()
            kwargs['S'] = S_k.copy()
            
        p_k = S_k.shape[0]
        kwargs['Omega_0'] = np.eye(p_k)
        kwargs['X_0'] = np.eye(p_k)
        estimates[k] = np.zeros((r,M,p_k,p_k))
        lowrank[k] = np.zeros((r,M,p_k,p_k))
        
        # start range search
        for j in np.arange(r):
            kwargs['lambda1'] = L[j]
            print(f"Reached row {j} of the grid/range")
            for m in np.arange(M):
                if latent:
                    kwargs['mu1'] = mu[m]
                    kwargs['latent'] = True
                            
                sol, info = ADMM_SGL(**kwargs)
                
                Theta_sol = sol['Theta']
                estimates[k][j,m,:,:] = Theta_sol.copy()
                
                if latent:
                    lowrank[k][j,m,:,:] = sol['L'].copy()
                    RANK[k,j,m] = np.linalg.matrix_rank(sol['L'])
                
                # warm start
                kwargs['Omega_0'] = sol['Omega'].copy()
                kwargs['X_0'] = sol['X'].copy()
                
                AIC[k,j,m] = aic_single(S_k, Theta_sol, N[k])
                for l in np.arange(len(gammas)):
                    BIC[l, k, j, m] = ebic_single(S_k, Theta_sol, N[k], gamma = gammas[l])
                    
                SP[k,j,m] = sparsity(Theta_sol)
                
    # get optimal low rank for each lambda
    tmpBIC = np.zeros((len(gammas), K, r))
    tmpBIC[:] = np.nan
    tmpAIC = np.zeros((K, r))
    tmpAIC[:] = np.nan
    
    ix_mu = np.zeros((K,r), dtype = int)
    
    # for each lambda, get optimal mu
    for k in np.arange(K):
        for j in np.arange(r):
            
            if method == 'AIC':
                ix_mu[k,j] = np.nanargmin(AIC[k,j,:])
                
            elif method == 'eBIC':
                ix_mu[k,j] = np.nanargmin(BIC[gamma_ix,k,j,:])
                
            tmpAIC[k,j] = AIC[k,j,ix_mu[k,j]]
            for l in np.arange(len(gammas)):
                tmpBIC[l,k,j] = BIC[l,k,j,ix_mu[k,j]]
        
    # get optimal lambda (uniform over k =1,..,K and individual)
    if method == 'AIC':
        tmpAIC[tmpAIC==-np.inf] = np.nan
        ix_uniform = np.nanargmin(tmpAIC.sum(axis=0))
        ix_indv = np.nanargmin(tmpAIC, axis = 1)
        
    elif method == 'eBIC':    
        tmpBIC[tmpBIC==-np.inf] = np.nan
        ix_uniform = np.nanargmin(tmpBIC[gamma_ix,:,:].sum(axis=0))
        ix_indv = np.nanargmin(tmpBIC[gamma_ix,:,:], axis = 1)
        
    # create the two estimators
    est_uniform = dict()
    est_indv = dict()
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
    
    statistics = {'BIC': BIC[gamma_ix,:,:,:], 'AIC': AIC, 'SP': SP, 'RANK': RANK, 'ix_uniform': ix_uniform, 'ix_indv': ix_indv, 'ix_mu': ix_mu}
    
    return est_uniform, est_indv, statistics

################################################################
    
def aic(S, Theta, N, L = None):
    """
    AIC information criterion after Danaher et al.
    excludes the diagonal
    """
    if type(S) == dict:
        aic = aic_dict(S, Theta, N, L)
    elif type(S) == np.ndarray:
        aic = aic_array(S, Theta, N, L)
    else:
        raise KeyError("Not a valid input type -- should be either dictionary or ndarray")
    
    return aic

def aic_dict(S, Theta, N, L = None):
    """
    S, Theta are dictionaries
    N is array of sample sizes
    """
    K = len(S.keys())
    if np.all(L is None):
        L = dict()
        for k in np.arange(K):
            L[k] = np.zeros(S[k].shape)     
    aic = 0
    for k in np.arange(K):
        aic += aic_single(S[k], Theta[k], N[k])
    return aic

def aic_array(S,Theta, N, L = None):
    (K,p,p) = S.shape
    
    if np.all(L is None):
        L = np.zeros((K,p,p))  
    if type(N) == int:
        N = np.ones(K) * N
    
    aic = 0
    for k in np.arange(K):
        aic += aic_single(S[k,:,:], Theta[k,:,:], N[k], L[k,:,:])

    return aic

def aic_single(S, Theta, N, L = None):
    (p,p) = S.shape
    
    if np.all(L is None):
        L = np.zeros((p,p))
        
    A = adjacency_matrix(Theta , t = 1e-5)
    aic = N*Sdot(S, Theta-L) - N*robust_logdet(Theta-L) + A.sum()
    
    return aic

################################################################
    
def ebic(S, Theta, N, gamma = 0.5, L = None):
    """
    extended BIC after Drton et al.
    """
    if type(S) == dict:
        ebic = ebic_dict(S, Theta, N, gamma, L)
    elif type(S) == np.ndarray:
        ebic = ebic_array(S, Theta, N, gamma, L)
    else:
        raise KeyError("Not a valid input type -- should be either dictionary or ndarray")
    
    return ebic

def ebic_single(S,Theta, N, gamma, L = None):
    (p,p) = S.shape
    
    if np.all(L is None):
        L = np.zeros((p,p))
        
    A = adjacency_matrix(Theta , t = 1e-5)
    bic = N*Sdot(S, Theta-L) - N*robust_logdet(Theta-L) + A.sum()/2 * (np.log(N)+ 4*np.log(p)*gamma)
    
    return bic

def ebic_array(S, Theta, N, gamma, L = None):
    (K,p,p) = S.shape
    if np.all(L is None):
        L = np.zeros((K,p,p))  
        
    if type(N) == int:
        N = np.ones(K) * N
        
    bic = 0
    for k in np.arange(K):
        bic += ebic_single(S[k,:,:], Theta[k,:,:], N[k], gamma, L[k,:,:])
    return bic

def ebic_dict(S, Theta, N, gamma, L = None):
    """
    S, Theta are dictionaries
    N is array of sample sizes
    """
    K = len(S.keys())
    if np.all(L is None):
        L = dict()
        for k in np.arange(K):
            L[k] = np.zeros(S[k].shape)    
            
    bic = 0
    for k in np.arange(K):
        bic += ebic_single(S[k], Theta[k], N[k], gamma, L[k])
        
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
    
    
def single_surface_plot(L1, L2, C, ax, name = 'eBIC'):
    
    #xx = (~np.isnan(C).any(axis=0))
    #L1 = L1[:,xx]
    #L2 = L2[:,xx]
    #C = C[:,xx]
    
    X = np.log10(L1)
    Y = np.log10(L2)
    Z = np.log(C)
    ax.plot_surface(X, Y, Z , cmap = plt.cm.ocean, linewidth=0, antialiased=True)
    
    #ax.set_xlabel('lambda_1')
    #ax.set_ylabel('lambda_2')
    ax.set_xlabel('w_1')
    ax.set_ylabel('w_2')
    ax.set_zlabel(name)
    ax.view_init(elev = 25, azim = 110)
    
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    
    return

def surface_plot(L1, L2, C, name = 'eBIC'):
    
    fig = plt.figure(figsize = (11,7))  
    if len(C.shape) == 2:
        ax = fig.gca(projection='3d')
        single_surface_plot(L1, L2, C, ax, name = name)
        
        
    else:
        for j in np.arange(C.shape[0]):
            ax = fig.add_subplot(2, 2, j+1, projection='3d')
            single_surface_plot(L1, L2, C[j,:,:], ax, name = name)
            ax.set_title('')
    
    return fig