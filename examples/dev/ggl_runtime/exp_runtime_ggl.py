"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""
from time import time
import numpy as np
from matplotlib import pyplot as plt


from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.ppdna_solver import PPDNA, warmPPDNA
from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix
from gglasso.helper.utils import get_K_identity
from gglasso.helper.experiment_helper import plot_runtime, discovery_rate

p = 100
K = 5
M = 10

reg = 'GGL'

Sigma, Theta = group_power_network(p, K, M, seed = 23456)

#%%
# runtime analysis ADMM vs. PPDNA on diff. sample sizes

f = np.array([0.1, 0.3, 0.7, 1.5])
vecN = (f * p).astype(int)

l1 = 2e-2 * np.ones(len(f))
l2 = 2e-2 * np.ones(len(f))

Omega_0 = get_K_identity(K,p)


TPR = np.zeros(len(vecN))
FPR = np.zeros(len(vecN))

iA = {}
iP = {}

for j in np.arange(len(vecN)):  
    
    S, sample = sample_covariance_matrix(Sigma, vecN[j], seed = 23456)
    
    solA, infoA = ADMM_MGL(S, l1[j], l2[j], reg , Omega_0 , max_iter = 2000, tol = 5e-5, stopping_criterion = 'kkt', verbose = False, measure = True)
    iA[j] = infoA
    
    TPR[j] = discovery_rate(solA['Theta'], Theta)['TPR']
    FPR[j] = discovery_rate(solA['Theta'], Theta)['FPR']
    
    #ppdna_params = {'sigma_fix': True, 'sigma_0' : 10.}
    ppdna_params = {}
    solP, infoP = warmPPDNA(S, l1[j], l2[j], reg, Omega_0, ppdna_params = ppdna_params,\
                            eps = 5e-5, eps_admm = 1e-2, verbose = True, measure = True)
    iP[j] = infoP

#%%
# plotting
save = False

plot_runtime(iA, iP, vecN, save = save)
    



   


  
    