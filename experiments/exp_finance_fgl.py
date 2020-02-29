"""
author: Fabian Schaipp

"""
from time import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.covariance import GraphicalLasso

from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.ppdna_solver import PPDNA, warmPPDNA
from gglasso.helper.experiment_helper import lambda_parametrizer, lambda_grid, discovery_rate, error
from gglasso.helper.experiment_helper import plot_evolution, plot_deviation, single_heatmap_animation
from gglasso.helper.model_selection import aic, ebic

from regain.covariance import LatentTimeGraphicalLasso, TimeGraphicalLasso

#df = pd.read_csv('../DaxConstituents.csv')
#df = df.drop(0, axis = 0)
#df.index = pd.to_datetime(df.Name)
#df.drop(columns = 'Name', inplace = True)
#df.columns = [t.split(' - ')[0] for t in df.columns]
#df.to_csv('data/dax_constituents.csv')

px = pd.read_csv('../data/dax_constituents.csv', index_col = 0)
px.index = pd.to_datetime(px.index)
ret = 100*np.log(px/px.shift(1)).dropna()

p = len(px.columns)
reg = 'FGL'
results = {}

#%%

def filter_by_year(ret):
    K = 10
    S = np.zeros((K,p,p))
    n = np.zeros(K)
    years = np.arange(start = 2003, stop = 2013)
    samples = {}
    for k in np.arange(K):
        dt = ret[ret.index.year == years[k]].values
        # demean to zero
        dt = dt - dt.mean(axis=0)
        samples[k] = dt.copy()
        n[k] = len(dt)
        S[k,:,:] = np.cov(dt, rowvar = False, bias = True)
    
    return S, samples, n, K

def filter_by_month(ret):
    K = 12
    S = np.zeros((K,p,p))
    n = np.zeros(K)
    ret = ret[ret.index.year == 2008]
    samples = {}
    for k in np.arange(K):
        samples_k = ret[ret.index.month == k+1].values
        samples[k] = samples_k
        n[k] = len(samples_k)
        S[k,:,:] = np.cov(samples_k, rowvar = False, bias = True)
    
    return S, samples, n, K

def filter_by_start(ret, start_date, K = 12, N = 22):
    """
    constructs one timestamp for each K intervals of N days 
    first interval starts at satrt_date
    """
    ret = ret[ret.index >= start_date]
    S = np.zeros((K,p,p))
    n = N * np.ones(K)
    samples = np.zeros((K,N,p))
    for k in np.arange(K):
        dt = ret.iloc[k*N:(k+1)*N, :]
        # demean to zero
        dt = dt - dt.mean(axis=0)
        samples[k,:,:] = dt
        S[k,:,:] = np.cov(samples[k], rowvar = False, bias = True)
        
    return S, samples, n

K = 13
N = 250
S, samples, n = filter_by_start(ret, start_date = '2002-01-02', K = K, N = N)

Sinv = np.linalg.pinv(S, hermitian = True)
#%%
L1, L2, _ = lambda_grid(num1 = 7, num2 = 3, reg = reg)
grid1 = L1.shape[0]; grid2 = L2.shape[1]

AIC = np.zeros((grid1, grid2))
BIC = np.zeros((grid1, grid2))

Omega_0 = np.zeros((K,p,p))
Theta_0 = np.zeros((K,p,p))

for g1 in np.arange(grid1):
    for g2 in np.arange(grid2):
        lambda1 = L1[g1,g2]
        lambda2 = L2[g1,g2]
        
        sol, info = ADMM_MGL(S,lambda1, lambda2, reg, Omega_0, eps_admm = 1e-3, verbose = False)
        Theta_sol = sol['Theta']
        Omega_sol = sol['Omega']
        
        # warm start
        Omega_0 = Omega_sol.copy()
        Theta_0 = Theta_sol.copy()
        
        AIC[g1,g2] = aic(S, Theta_sol, n.mean())
        BIC[g1,g2] = ebic(S, Theta_sol, n.mean(), gamma = 0.1)

ix= np.unravel_index(np.nanargmin(BIC), BIC.shape)
ix2= np.unravel_index(np.nanargmin(AIC), AIC.shape)
lambda1 = L1[ix]
lambda2 = L2[ix]

print("Optimal lambda values: (l1,l2) = ", (lambda1,lambda2))


#%%
singleGL = GraphicalLasso(alpha = 1.5*lambda1, tol = 1e-2, max_iter = 4000, verbose = True)

res = np.zeros((K,p,p))
for k in np.arange(K):
    #model = quic.fit(S[k,:,:], verbose = 1)
    model = singleGL.fit(samples[k,:,:])
    
    res[k,:,:] = model.precision_

results['GLASSO'] = {'Theta' : res}

#%%
start = time()
sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, rho = 1, max_iter = 100, \
                                                        eps_admm = 1e-5, verbose = True, measure = True)
end = time()

print(f"Running time for ADMM was {end-start} seconds")

results['ADMM'] = {'Theta' : sol['Theta']}

#%%
alpha = lambda1 * N
beta = lambda2 * N
tau = 1 * N
eta = .1 * N
ltgl = LatentTimeGraphicalLasso(alpha = alpha, beta = beta, tau = tau, eta = eta, psi = 'l1', phi = 'l1',\
                                rho = 1, tol = 1e-4, max_iter=2000, verbose = True)

ltgl = ltgl.fit(samples)
print(np.linalg.matrix_rank(ltgl.latent_))
results['LGTL'] = {'Theta' : ltgl.precision_, 'L' : ltgl.latent_}
#%%
plot_deviation(results, latent = results.get('LGTL').get('L'))

single_heatmap_animation(results.get('LGTL').get('Theta'), method = 'ADMM', save = False)



