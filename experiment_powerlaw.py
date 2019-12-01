"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


from gglasso.solver.ggl_solver import PPDNA
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, aic, error, draw_group_graph


p = 10
K = 3
N = 9
M = 2


Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)/np.linalg.norm(np.eye(p))

draw_group_graph(Theta)

S = sample_covariance_matrix(Sigma, N)
Sinv = np.linalg.pinv(S, hermitian = True)

#%%
# grid search for best lambda values
L1, L2 = lambda_grid(num1 = 7, num2 = 2)
grid1 = L1.shape[0]; grid2 = L2.shape[1]

ERR = np.zeros((grid1, grid2))
FPR = np.zeros((grid1, grid2))
TPR = np.zeros((grid1, grid2))
AIC = np.zeros((grid1, grid2))

for g1 in np.arange(grid1):
    for g2 in np.arange(grid2):
        lambda1 = L1[g1,g2]
        lambda2 = L2[g1,g2]
    
        Omega_0 = np.zeros((K,p,p))
        #Omega_0 = np.linalg.pinv(S)
        reg = 'FGL'
        
        sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = None, sigma_0 = 10, max_iter = 20, \
                                                    eps_ppdna = 1e-5 , verbose = True)
        
        Theta_sol = sol['Theta']
        
        TPR[g1,g2] = discovery_rate(Theta_sol, Theta, t = 1e-5)['TPR']
        FPR[g1,g2] = discovery_rate(Theta_sol, Theta, t = 1e-5)['FPR']
        ERR[g1,g2] = error(Theta_sol, Theta)
        AIC[g1,g2] = aic(S,Theta_sol,N)

#%%
# plot results

plt.figure()
plt.plot(FPR, TPR)

fig,axs = plt.subplots(nrows = 1, ncols = 3)
sns.heatmap(TPR, annot = True, ax = axs[0])
sns.heatmap(ERR, annot = True, ax = axs[1])
sns.heatmap(AIC, annot = True, ax = axs[2])


#%%
# optimize with optimal lambda values

lambda1 = 0.2
lambda2 = 0.1

