"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""
from time import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.ggl_solver import PPDNA, warmPPDNA
from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import get_K_identity, lambda_grid, discovery_rate, aic, ebic, error
from gglasso.helper.experiment_helper import draw_group_heatmap

p = 100
K = 5
M = 10

reg = 'GGL'
save = True

Sigma, Theta = group_power_network(p, K, M)

draw_group_heatmap(Theta, save = save)


#%%
# runtime analysis ADMM vs. PPDNA on diff. sample sizes

f = np.array([0.5, 0.9, 2, 20])
vecN = (f * p).astype(int)

l1 = 2e-2 * np.ones(len(f))
l2 = 2e-2 * np.ones(len(f))

Omega_0 = get_K_identity(K,p)

RT_ADMM = np.zeros(len(vecN))
RT_PPA = np.zeros(len(vecN))
TPR = np.zeros(len(vecN))
FPR = np.zeros(len(vecN))

iA = {}
iP = {}

for j in np.arange(len(vecN)):  
    
    S, sample = sample_covariance_matrix(Sigma, vecN[j])
    
    #start = time()
    solA, infoA = ADMM_MGL(S, l1[j], l2[j], reg , Omega_0 , eps_admm = 5e-5, verbose = False, measure = True)
    #end = time()
    #RT_ADMM[j] = end-start
    iA[j] = infoA
    
    TPR[j] = discovery_rate(solA['Theta'], Theta)['TPR']
    FPR[j] = discovery_rate(solA['Theta'], Theta)['FPR']
    
    #start = time()
    solP, infoP = warmPPDNA(S, l1[j], l2[j], reg, Omega_0, eps = 5e-5, eps_admm = 1e-3, verbose = False, measure = True)
    #end = time()
    #RT_PPA[j] = end-start
    iP[j] = infoP

#%%
    
color_dict = get_default_color_coding()
fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15,8)) 

for j in np.arange(len(vecN)):
    
    ax = axs.reshape(-1)[j]
    with sns.axes_style("whitegrid"):
        
        p1 = ax.plot(iA[j]['kkt_residual'], c = color_dict['ADMM'], label = 'ADMM residual')
        p2 = ax.plot(iP[j]['kkt_residual'], c = color_dict['PPDNA'], marker = 'o', markersize = 3, label = 'PPDNA residual')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(1e-6,0.2)
        
        ax2 = ax.twinx()
        ax2.set_xscale('log')
        p3 = ax2.plot(iA[j]['runtime'].cumsum(), linestyle = '--', c = color_dict['ADMM'], alpha = 0.7, label = 'ADMM runtime')
        p4 = ax2.plot(iP[j]['runtime'].cumsum(), linestyle = '--', c = color_dict['PPDNA'], marker = 'o', markersize = 3, alpha = 0.7, label = 'PPDNA runtime')
        
        ax.vlines(iP[j]['iter_admm'], 0, 0.2, 'grey')
        ax.set_xlim(iP[j]['iter_admm'] - 5, )
        
        if j in [0,2]:
            ax.set_ylabel('KKT residual')
        if j in [1,3]:
            ax2.set_ylabel('Cumulated runtime [sec]')
        if j in [2,3]:
            ax.set_xlabel('Iteration number')
        
        ax.set_title(f'Sample size = {vecN[j]}')
        
        lns = p1+p2+p3+p4
        labs = [l.get_label() for l in lns]
        fig.legend(lns, labs, loc=0)
        
    path_rt = 'plots//ggl_runtime//'  
    fig.savefig(path_rt + 'runtimeN.pdf', dpi = 300)
        
#%%
        
vecP = np.array([20, 100, 200, 1000])  



   


  
    