"""
Group Graphical Lasso experiment
=================================

We investigate the recovery performance of Group Graphical Lasso on Powerlaw networks, compared to estimating the precision matrices independently with SGL.

We generate a precision matrix with block-wise powerlaw networks. In each instance, one of the blocks is randomly set to zero. Hence, the true underlying precision matrices are group sparse.  
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.covariance import GraphicalLasso

from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, error
from gglasso.helper.utils import get_K_identity
from gglasso.helper.experiment_helper import draw_group_heatmap, plot_fpr_tpr, plot_diff_fpr_tpr, surface_plot
from gglasso.helper.model_selection import aic, ebic

p = 100
K = 5
N = 80
M = 10

reg = 'GGL'

Sigma, Theta = group_power_network(p, K, M, scale = False, nxseed = 2340)

S, sample = sample_covariance_matrix(Sigma, N)

# %%
#  Parameter selection (GGL)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We do a grid search over :math:`\lambda_1` and :math:`\lambda_2` values.
# On each grid point we evaluate True/False Discovery Rate (TPR/FPR), True/False Discovery of Differential edges and AIC and eBIC.
#
# Note: the package contains functions for doing this grid search, but here we also want to evaluate True and False positive rates on each grid points.
#
#

L1, L2, W2 = lambda_grid(num1 = 3, num2 = 10, reg = reg)
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
X_0 = np.zeros((K,p,p))

for g2 in np.arange(grid2):
    for g1 in np.arange(grid1):
        lambda1 = L1[g1,g2]
        lambda2 = L2[g1,g2]
              
        sol, info =  ADMM_MGL(S, lambda1, lambda2, reg , Omega_0, Theta_0 = Theta_0, X_0 = X_0, tol = 1e-8, rtol = 1e-8, verbose = False, measure = False)

        Theta_sol = sol['Theta']
        Omega_sol = sol['Omega']
        X_sol = sol['X']
        
        # warm start
        Omega_0 = Omega_sol.copy()
        Theta_0 = Theta_sol.copy()
        X_0 = X_sol.copy()
        
        dr = discovery_rate(Theta_sol, Theta)
        TPR[g1,g2] = dr['TPR']
        FPR[g1,g2] = dr['FPR']
        DTPR[g1,g2] = dr['TPR_DIFF']
        DFPR[g1,g2] = dr['FPR_DIFF']
        ERR[g1,g2] = error(Theta_sol, Theta)
        AIC[g1,g2] = aic(S, Theta_sol, N)
        BIC[g1,g2] = ebic(S, Theta_sol, N, gamma = 0.1)
        
            

# get optimal lambda
ix= np.unravel_index(np.nanargmin(BIC), BIC.shape)
ix2= np.unravel_index(np.nanargmin(AIC), AIC.shape)

l1opt = L1[ix]
l2opt = L2[ix]

print("Optimal lambda values: (l1,l2) = ", (l1opt,l2opt))

# %%
#  Solving group sparse problems with SGL
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We now solve K independent SGL problems and find the best :math:`\lambda_1` parameter.
#
#

ALPHA = np.logspace(start = 0, stop = -1.5, num = 15, base = 10)

FPR_GL = np.zeros(len(ALPHA))
TPR_GL = np.zeros(len(ALPHA))
DFPR_GL = np.zeros(len(ALPHA))
DTPR_GL = np.zeros(len(ALPHA))

for a in np.arange(len(ALPHA)):
    singleGL = GraphicalLasso(alpha = ALPHA[a], tol = 1e-4, max_iter = 50, verbose = False)
    singleGL_sol = np.zeros((K,p,p))
    for k in np.arange(K):
        model = singleGL.fit(sample[k,:,:].T)
        singleGL_sol[k,:,:] = model.precision_
    
    dr = discovery_rate(singleGL_sol, Theta)
    TPR_GL[a] = dr['TPR']
    FPR_GL[a] = dr['FPR']
    DTPR_GL[a] = dr['TPR_DIFF']
    DFPR_GL[a] = dr['FPR_DIFF']
    

# %%
# Solving
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To demonstrate again how to call the ADMM solver, we solve to high accuracy again for the best values of :math:`\lambda_1` and :math:`lambda_2`.
#

Omega_0 = get_K_identity(K,p)
solA, infoA = ADMM_MGL(S, l1opt, l2opt, reg , Omega_0, tol = 1e-10, rtol = 1e-10, verbose = True, measure = True)


# %%
# Plotting: TPR, FPR, differential edges
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We plot FPR and TPR for all grid points for GGL and SGL. 
# The circle-shape marker marks the point which would have been selected by eBIC. The diamond-shaped marker marks the point selected by AIC.
# 
# Differential edges are edges which are present in at least one but not all of the K precision matrices.
   
fig,ax = plot_fpr_tpr(FPR, TPR, ix, ix2, FPR_GL, TPR_GL, W2)
ax.set_xlim(-0.01, 0.1)
ax.set_ylim(0.3,1)

fig,ax = plot_diff_fpr_tpr(DFPR, DTPR, ix, ix2, DFPR_GL, DTPR_GL, W2)

