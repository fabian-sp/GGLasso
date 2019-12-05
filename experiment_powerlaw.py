"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""
from time import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


from gglasso.solver.ggl_solver import PPDNA
from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, aic, error, draw_group_graph


p = 30
K = 5
N = 200
M = 3

reg = 'GGL'

if reg == 'GGL':
    Sigma, Theta = group_power_network(p, K, M)
elif reg == 'FGL':
    Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)/np.linalg.norm(np.eye(p))

draw_group_graph(Theta)

S = sample_covariance_matrix(Sigma, N)
Sinv = np.linalg.pinv(S, hermitian = True)

#%%
# grid search for best lambda values with warm starts
L1, L2 = lambda_grid(num1 = 5, num2 = 2, reg = reg)
grid1 = L1.shape[0]; grid2 = L2.shape[1]

ERR = np.zeros((grid1, grid2))
FPR = np.zeros((grid1, grid2))
TPR = np.zeros((grid1, grid2))
AIC = np.zeros((grid1, grid2))

Omega_0 = np.zeros((K,p,p))
Theta_0 = np.zeros((K,p,p))

for g1 in np.arange(grid1):
    for g2 in np.arange(grid2):
        lambda1 = L1[g1,g2]
        lambda2 = L2[g1,g2]
    
        
        sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, sigma_0 = 10, max_iter = 20, \
                                                    eps_ppdna = 1e-3 , verbose = False)
        
        Theta_sol = sol['Theta']
        Omega_sol = sol['Omega']
        
        # warm start
        Omega_0 = Omega_sol.copy()
        Theta_0 = Theta_sol.copy()
        
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
# accuracy impact on total error analysis
L1, L2 = lambda_grid(num1 = 1, num2 = 5, reg = reg)
grid1 = L1.shape[0]
grid2 = 7

EPS = np.logspace(start = -.5, stop = -5, num = grid2, base = 10)

ERR = np.zeros((grid1, grid2))
AIC = np.zeros((grid1, grid2))
RT = np.zeros((grid1, grid2))

#Omega_0 = np.zeros((K,p,p)); Theta_0 = np.zeros((K,p,p)); X_0 = np.zeros((K,p,p))
Om_0_0 = np.zeros((K,p,p)); Th_0_0 = np.zeros((K,p,p)); X_0_0 = np.zeros((K,p,p))

for g1 in np.arange(grid1):
    
    Omega_0 = Om_0_0.copy()
    Theta_0 = Th_0_0.copy()
    X_0 = X_0_0.copy()
    
    for g2 in np.arange(grid2):
            
        start = time()
        sol, info = PPDNA(S, L1[g1], L2[g1], reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, sigma_0 = 10, max_iter = 20, \
                                                        eps_ppdna = EPS[g2] , verbose = False)
        end = time()
        
        Theta_sol = sol['Theta']
        Omega_sol = sol['Omega']
        X_sol = sol['X']
        
        # warm start, has to be resetted whenever changing g1 
        Omega_0 = Omega_sol.copy(); Theta_0 = Theta_sol.copy(); X_0 = X_sol.copy()
        if g2 == 0:
            Om_0_0 = Omega_sol.copy(); Th_0_0 = Theta_sol.copy(); X_0_0 = X_sol.copy()
        
        ERR[g1,g2] = error(Theta_sol, Theta)
        AIC[g1,g2] = aic(S,Theta_sol,N)
        RT[g1,g2] = end-start


#%%
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#
#X,Y = np.meshgrid(EPS, L2)
#Z = AIC
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)    
pal = sns.color_palette("GnBu_d", grid1)

sns.set()
fig, ax = plt.subplots(1,1)
for l in np.arange(grid1):
    ax.plot(EPS, ERR[l,:], c=pal[l] )

ax.set_xlim(EPS.max(), EPS.min())
ax.legend(labels = L2)
ax.set_xscale('log')