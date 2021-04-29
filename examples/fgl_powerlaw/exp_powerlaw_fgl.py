"""
author: Fabian Schaipp

Set the working directory to the file location if you want to save the plots.

This is a script for investigating Fused Graphical Lasso on Powerlaw networks.
Sigma denotes the true covariance matrix, Theta the true precision matrix.
"""
from time import time
import numpy as np

from sklearn.covariance import GraphicalLasso

from gglasso.solver.ppdna_solver import warmPPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, error
from gglasso.helper.utils import get_K_identity, deviation
from gglasso.helper.experiment_helper import plot_evolution, plot_deviation, surface_plot, multiple_heatmap_animation, single_heatmap_animation
from gglasso.helper.model_selection import aic, ebic

from regain.covariance import TimeGraphicalLasso

p = 100
K = 10
N = 5000
M = 5
L = int(p/M)

reg = 'FGL'

Sigma, Theta = time_varying_power_network(p, K, M, nxseed = 2340)

#single_heatmap_animation(Theta)

S, sample = sample_covariance_matrix(Sigma, N)
S_train, sample_train = sample_covariance_matrix(Sigma, N)
Sinv = np.linalg.pinv(S, hermitian = True)


results = {}
results['truth'] = {'Theta' : Theta}

#%% grid search for best lambda values with warm starts

L1, L2, _ = lambda_grid(num1 = 10, num2 = 5, reg = reg)
grid1 = L1.shape[0]; grid2 = L2.shape[1]

ERR = np.zeros((grid1, grid2))
FPR = np.zeros((grid1, grid2))
TPR = np.zeros((grid1, grid2))
DFPR = np.zeros((grid1, grid2))
DTPR = np.zeros((grid1, grid2))
AIC = np.zeros((grid1, grid2))
BIC = np.zeros((grid1, grid2))

Omega_0 = get_K_identity(K,p)
Theta_0 = get_K_identity(K,p)

for g1 in np.arange(grid1):
    for g2 in np.arange(grid2):
        lambda1 = L1[g1,g2]
        lambda2 = L2[g1,g2]
              
        sol, info = warmPPDNA(S_train, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, eps = 1e-3, verbose = False, measure = False)
        
        #sol, info =  ADMM_MGL(S_train, lambda1, lambda2, reg , Omega_0, Theta_0 = Theta_0, tol = 1e-10, rtol = 1e-10, verbose = False, measure = False)

        Theta_sol = sol['Theta']
        Omega_sol = sol['Omega']
        
        # warm start
        Omega_0 = Omega_sol.copy()
        Theta_0 = Theta_sol.copy()
        
        TPR[g1,g2] = discovery_rate(Theta_sol, Theta)['TPR']
        FPR[g1,g2] = discovery_rate(Theta_sol, Theta)['FPR']
        DTPR[g1,g2] = discovery_rate(Theta_sol, Theta)['TPR_DIFF']
        DFPR[g1,g2] = discovery_rate(Theta_sol, Theta)['FPR_DIFF']
        ERR[g1,g2] = error(Theta_sol, Theta)
        AIC[g1,g2] = aic(S_train, Theta_sol, N)
        BIC[g1,g2] = ebic(S_train, Theta_sol, N, gamma = 0.1)

# get optimal lambda
ix= np.unravel_index(np.nanargmin(BIC), BIC.shape)
ix2= np.unravel_index(np.nanargmin(AIC), AIC.shape)
lambda1 = L1[ix]
lambda2 = L2[ix]

print("Optimal lambda values: (l1,l2) = ", (lambda1,lambda2))


#%% solve with scikit SGL and eBIC selection

#lambda1 + (1/np.sqrt(2)) *lambda2
ALPHA = 2*np.logspace(start = -3, stop = -1, num = 10, base = 10)
SGL_BIC = np.zeros(len(ALPHA))
all_res = list()

for j in range(len(ALPHA)):
    res = np.zeros((K,p,p))
    singleGL = GraphicalLasso(alpha = ALPHA[j], tol = 1e-6, max_iter = 200, verbose = False)
    for k in np.arange(K):
        model = singleGL.fit(sample_train[k,:,:].T)
        res[k,:,:] = model.precision_
    
    all_res.append(res)
    SGL_BIC[j] = ebic(S_train, res, N, gamma = 0.1)

ix_SGL = np.argmin(SGL_BIC)
results['SGL'] = {'Theta' : all_res[ix_SGL]}

#%% solve with PPDNA (first execution is typically slow due to numba compilation)

Omega_0 = results.get('SGL').get('Theta')
Theta_0 = Omega_0.copy()
X_0 = np.zeros((K,p,p))

start = time()
sol, info = warmPPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, eps = 1e-4 , verbose = True, measure = True)
end = time()

print(f"Running time for PPDNA was {end-start} seconds")
results['PPDNA'] = {'Theta' : sol['Theta']}

#%% solve with ADMM

start = time()
sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, rho = 1, max_iter = 1000, \
                                                        tol = 1e-7, rtol = 1e-7, verbose = True, measure = True)
end = time()

print(f"Running time for ADMM was {end-start} seconds")

results['ADMM'] = {'Theta' : sol['Theta']}

#%% solve with regain
# regain needs data in format (N*K,p)
# regain has TV penalty also on the diagonal, hence results may be slightly different than ADMM_MGL

tmp = sample.transpose(1,0,2).reshape(p,-1).T

start = time()
alpha = N*lambda1
beta = N*lambda2 
ltgl = TimeGraphicalLasso(alpha = alpha, beta = beta , psi = 'l1', \
                          rho = 1., tol = 1e-5, rtol = 1e-5,  max_iter = 2000, verbose = False)
ltgl = ltgl.fit(X = tmp, y = np.repeat(np.arange(K),N))
end = time()

print(f"Running time for LTGL was {end-start}  seconds")

results['LTGL'] = {'Theta' : ltgl.precision_}

#%% plotting

Theta_admm = results.get('ADMM').get('Theta')
Theta_ppdna = results.get('PPDNA').get('Theta')
Theta_ltgl = results.get('LTGL').get('Theta')
Theta_sgl = results.get('SGL').get('Theta')


print("Norm(Regain-ADMM)/Norm(ADMM):", np.linalg.norm(Theta_ltgl - Theta_admm)/ np.linalg.norm(Theta_admm))
print("Norm(PPDNA-ADMM)/Norm(ADMM):", np.linalg.norm(Theta_ppdna - Theta_admm)/ np.linalg.norm(Theta_admm))

# whether to save the plots as pdf-files
save = False

fig = surface_plot(L1, L2, BIC, name = 'eBIC')
if save:
    fig.savefig('../plots/fgl_powerlaw/surface.pdf', dpi = 500)

plot_evolution(results, block = 0, L = L, save = save)

plot_evolution(results, block = 2, L = L, save = save)

del results['PPDNA']

plot_deviation(results, save = save)


# animate truth and solution
single_heatmap_animation(Theta_sgl, method = 'SGL', save = False)
multiple_heatmap_animation(Theta, results, save = False)



