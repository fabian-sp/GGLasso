"""
author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .basic_linalg import Sdot, adjacency_matrix
from .experiment_helper import mean_sparsity, sparsity, consensus

from .experiment_helper import get_K_identity as id_array
from .ext_admm_helper import get_K_identity as id_dict
from ..solver.single_admm_solver import ADMM_SGL


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

def grid_search(solver, S, N, p, reg, l1, method= 'eBIC', l2 = None, w2 = None, G = None):
    """
    method for doing model selection using grid search and AIC/eBIC
    we work the grid columnwise, i.e. hold l1 constant and change l2
    """
    
    assert method in ['AIC', 'eBIC']
    assert reg in ['FGL', 'GGL']
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
    
    UQED = np.zeros((grid1, grid2))
    
    kwargs = {'reg': reg, 'S': S, 'eps_admm': 1e-3, 'verbose': True, 'measure': True}
    if type(S) == dict:
        K = len(S.keys())
        Omega_0 = id_dict(p)
        kwargs['G'] = G
    elif type(S) == np.ndarray:
        K = S.shape[0]
        Omega_0 = id_array(K,p)
        
    kwargs['Omega_0'] = Omega_0.copy()
    
    curr_min = np.inf
    curr_best = None
    # run down the columns --> hence move g1 fastest
    for g2 in np.arange(grid2):
        for g1 in np.arange(grid1):
      
            print("Current grid point: ", (L1[g1,g2],L2[g1,g2]) )
            if SKIP[g1,g2]:
                print("SKIP")
                continue
            kwargs['lambda1'] = L1[g1,g2]
            kwargs['lambda2'] = L2[g1,g2]

            sol, info = solver(**kwargs)
            Omega_sol = sol['Omega']
            Theta_sol = sol['Theta']
            
            #if mean_sparsity(Theta_sol) >= 0.18:
            #    SKIP[g1:, g2:] = True
            
            # warm start
            kwargs['Omega_0'] = Omega_sol.copy()
            kwargs['X0'] = sol['X0'].copy()
            kwargs['X1'] = sol['X1'].copy()
            
            AIC[g1,g2] = aic(S, Theta_sol, N)
            for j in np.arange(len(gammas)):
                BIC[j, g1,g2] = ebic(S, Theta_sol, N, gamma = gammas[j])
                
            SP[g1,g2] = mean_sparsity(Theta_sol)
            
            if np.all(G!= None):
                nnz = consensus(Theta_sol, G)
                UQED[g1,g2] = (nnz >= 1).sum()
                
            
            print("Current eBIC grid:")
            print(BIC)
            print("Current Sparsity grid:")
            print(SP)
            
            if BIC[gamma_ix,g1,g2].mean() < curr_min:
                curr_min = BIC[gamma_ix,g1,g2]
                curr_best = sol.copy()
    
    # get optimal lambda
    if method == 'AIC':
        AIC[AIC==-np.inf] = np.nan
        ix= np.unravel_index(np.nanargmin(AIC), AIC.shape)
    elif method == 'eBIC':    
        BIC[BIC==-np.inf] = np.nan
        ix= np.unravel_index(np.nanargmin(BIC[gamma_ix,:,:]), BIC[gamma_ix,:,:].shape)
    return AIC, BIC, L1, L2, ix, SP, UQED, curr_best

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
    tmpBIC = np.zeros((len(gammas), K, r, M))
    tmpBIC[:] = np.nan
    
    tmpAIC = np.zeros((K, r, M))
    tmpAIC[:] = np.nan
    
    tmpSP = np.zeros((K, r, M))
    tmpSP[:] = np.nan
    
    estimates = dict()
    
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
        
        # start range search
        for m in np.arange(M):
            if latent:
                kwargs['mu'] = mu[m]
                
            for j in np.arange(r):
                kwargs['lambda1'] = L[j]
                sol, info = ADMM_SGL(**kwargs)
                
                Theta_sol = sol['Theta']
                estimates[k][j,m,:,:] = Theta_sol.copy()
            
                # warm start
                kwargs['Omega_0'] = sol['Omega'].copy()
                kwargs['X_0'] = sol['X'].copy()
                
                tmpAIC[k,j,m] = aic_single(S_k, Theta_sol, N[k])
                for l in np.arange(len(gammas)):
                    tmpBIC[l, k, j, m] = ebic_single(S_k, Theta_sol, N[k], gamma = gammas[l])
                    
                tmpSP[k,j,m] = sparsity(Theta_sol)
                
    # get optimal low rank for each lambda
    if M > 1:
        BIC = np.zeros((len(gammas), K, r))
        BIC[:] = np.nan
        AIC = np.zeros((K, r))
        AIC[:] = np.nan
       
        for k in np.arange(K):
            for j in np.arange(r):
                ixx = np.nanargmin(tmpAIC[k,j,:])
                AIC[k,j] = tmpAIC[k,j,ixx]
                for l in np.arange(len(gammas)):
                    ixx = np.nanargmin(tmpBIC[l,k,j,:])
                    BIC[l,k,j] = tmpBIC[l,k,j,ixx]
    else:
        BIC = tmpBIC.squeeze()
        AIC = tmpAIC.squeeze()     
        
    SP = tmpSP.copy()
    # get optimal lambda
    if method == 'AIC':
        AIC[AIC==-np.inf] = np.nan
        ix_uniform = np.nanargmin(AIC.sum(axis=0))
        ix_indv = np.nanargmin(AIC, axis = 1)
        
    elif method == 'eBIC':    
        BIC[BIC==-np.inf] = np.nan
        ix_uniform = np.nanargmin(BIC[gamma_ix,:,:].sum(axis=0))
        ix_indv = np.nanargmin(BIC[gamma_ix,:,:], axis = 1)
        
    # create the two estimators
    est_uniform = dict()
    est_indv = dict()
    for k in np.arange(K):
        est_uniform[k] = estimates[k][ix_uniform,:,:]
        est_indv[k] = estimates[k][ix_indv[k], :, :]
        
    return AIC, BIC, SP, est_uniform, est_indv, ix_uniform


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

def ebic(S, Theta, N, gamma = 0.5):
    """
    extended BIC after Drton et al.
    """
    if type(S) == dict:
        aic = ebic_dict(S, Theta, N, gamma)
    elif type(S) == np.ndarray:
        aic = ebic_array(S, Theta, N, gamma)
    else:
        raise KeyError("Not a valid input type -- should be either dictionary or ndarray")
    
    return aic

def aic_array(S,Theta, N):
    (K,p,p) = S.shape
    
    if type(N) == int:
        N = np.ones(K) * N
    
    A = adjacency_matrix(Theta , t = 1e-5)
    nonzero_count = A.sum(axis=(1,2))/2
    aic = 0
    for k in np.arange(K):
        aic += N[k]*Sdot(S[k,:,:], Theta[k,:,:]) - N[k]*robust_logdet(Theta[k,:,:]) + 2*nonzero_count[k]
        
    return aic

def aic_single(S,Theta, N):
    (p,p) = S.shape
    A = adjacency_matrix(Theta , t = 1e-5)
    aic = N*Sdot(S, Theta) - N*robust_logdet(Theta) + A.sum()
    
    return aic


def ebic_single(S,Theta, N, gamma):
    (p,p) = S.shape
    A = adjacency_matrix(Theta , t = 1e-5)
    bic = N*Sdot(S, Theta) - N*robust_logdet(Theta) + A.sum()/2 * (np.log(N)+ 4*np.log(p)*gamma)
    
    return bic
    

def ebic_array(S, Theta, N, gamma):
    (K,p,p) = S.shape
    if type(N) == int:
        N = np.ones(K) * N
    
    A = adjacency_matrix(Theta , t = 1e-5)
    nonzero_count = A.sum(axis=(1,2))/2
    
    bic = 0
    for k in np.arange(K):
        bic += N[k]*Sdot(S[k,:,:], Theta[k,:,:]) - N[k]*robust_logdet(Theta[k,:,:]) + nonzero_count[k] * (np.log(N[k])+ 4*np.log(p)*gamma)
    
    return bic


def ebic_dict(S, Theta, N, gamma):
    """
    S, Theta are dictionaries
    N is array of sample sizes
    """
    K = len(S.keys())
    bic = 0
    for k in np.arange(K):
        A = adjacency_matrix(Theta[k] , t = 1e-5)
        p = S[k].shape[0]
        bic += N[k]*Sdot(S[k], Theta[k]) - N[k]*robust_logdet(Theta[k]) + A.sum()/2 * (np.log(N[k])+ 4*np.log(p)*gamma)
        
    return bic
        

def aic_dict(S, Theta, N):
    """
    S, Theta are dictionaries
    N is array of sample sizes
    """
    K = len(S.keys())
    aic = 0
    for k in np.arange(K):
        A = adjacency_matrix(Theta[k] , t = 1e-5)
        aic += N[k]*Sdot(S[k], Theta[k]) - N[k]*robust_logdet(Theta[k]) + A.sum()
        
    return aic

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

def surface_plot(L1, L2, C, name = 'eBIC', save = False):
    
    fig = plt.figure(figsize = (11,7))  
    if len(C.shape) == 2:
        ax = fig.gca(projection='3d')
        single_surface_plot(L1, L2, C, ax, name = name)
        
        
    else:
        for j in np.arange(C.shape[0]):
            ax = fig.add_subplot(2, 2, j+1, projection='3d')
            single_surface_plot(L1, L2, C[j,:,:], ax, name = name)
            ax.set_title('')
    
    if save:
        fig.savefig('data/slr_results/res_multiple/surface.png', dpi = 500)
        
    return