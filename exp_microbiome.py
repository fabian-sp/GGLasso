"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from microbiome_helper import load_and_transform
from gglasso.solver.ext_admm_solver import ext_ADMM_MGL
from gglasso.solver.single_admm_solver import ADMM_SGL

from gglasso.helper.experiment_helper import sparsity
from gglasso.helper.ext_admm_helper import get_K_identity, check_G, load_G, save_G
from gglasso.helper.model_selection import grid_search, single_range_search, ebic, surface_plot

K = 26
reg = 'GGL'

all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 5, compute_G = False)

#save_G('data/slr_results/', G)
G = load_G('data/slr_results/')
check_G(G, p)

#%%


l1 = np.linspace(1, 0.4, 3)
l1 = np.append(l1, np.linspace(0.2, 0.05, 5))
w2 = np.logspace(-1, -5, 4)

AIC, BIC, L1, L2, ix, SP, SKIP, sol1 = grid_search(ext_ADMM_MGL, S, num_samples, p, reg, l1 = l1, method = 'eBIC', w2 = w2, G = G)

res_multiple = sol1['Theta']
surface_plot(L1,L2, BIC, save = False)


#%%

#l1 = np.linspace(0.2, 0.05, 5)
#l1 = 5*np.logspace(-1, -2.5, 6)

sAIC, sBIC, sSP, sol2, sol3, ix_uniform = single_range_search(S, l1, num_samples)


#%%
#L = G.shape[1]
#groupsize = (G!=-1).sum(axis=2)[0]
#
#nnz = np.zeros(L)
#Theta = sol['Theta']
#for l in np.arange(L):
#    for k in np.arange(K):
#        if G[0,l,k] == -1:
#            continue
#        else:
#            nnz[l] += abs(Theta[k][G[0,l,k], G[1,l,k]]) >= 1e-5
#    
    
#%%

Omega_0 = get_K_identity(p)
lambda1 = 0.1625
lambda2 = 3e-6
sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True)

res_multiple = sol['Theta']


#%%
# section for saving results
def save_result(res, typ):
    path = 'data/slr_results/res_' + typ
    K = len(res.keys())
    for k in np.arange(K):
        res_k = pd.DataFrame(res[k], index = all_csv[k].index, columns = all_csv[k].index)
        res_k.to_csv(path + '/theta_' + str(k+1) + ".csv")
    print("All files saved")
    return

save_result(res_multiple, 'multiple')

for j in np.arange(BIC.shape[0]):
    np.savetxt('data/slr_results/res_multiple/BIC_' + str(j) + '.csv', BIC[j,:,:])
np.savetxt('data/slr_results/res_multiple/AIC.csv', AIC)
np.savetxt('data/slr_results/res_multiple/SP.csv', SP)
np.savetxt('data/slr_results/res_multiple/L1.csv', L1)
np.savetxt('data/slr_results/res_multiple/L2.csv', L2)


#%%
save_result(sol2, 'single_unif')
save_result(sol3, 'single')

for j in np.arange(sBIC.shape[0]):
    np.savetxt('data/slr_results/res_single/BIC_' + str(j) + '.csv', sBIC[j,:,:])
np.savetxt('data/slr_results/res_single/AIC.csv', sAIC)
np.savetxt('data/slr_results/res_single/SP.csv', sSP)


#%%
# section for loading results
AIC = np.loadtxt('data/slr_results/AIC.csv')
SP = np.loadtxt('data/slr_results/SP.csv')
L1 = np.loadtxt('data/slr_results/L1.csv')
L2 = np.loadtxt('data/slr_results/L2.csv')

BIC = np.zeros((4, L1.shape[0], L1.shape[1]))
for j in np.arange(4):
    BIC[j,:,:] = np.loadtxt('data/slr_results/BIC_' + str(j) + '.csv')
    
    
#%%
########## EVALUATION ########################    
  
info = pd.DataFrame(index = np.arange(K)+1)
info['samples'] = num_samples
info['OTUs'] = p
info['group entry ratio'] = (G[1,:,:] != -1).sum(axis=0) / (p*(p-1)/2)

info['sparsity GGL'] = [sparsity(sol1['Theta'][k]) for k in sol1['Theta'].keys()]
info['sparsity single/uniform'] = [sparsity(sol2[k]) for k in sol2.keys()]
info['sparsity single/indv'] = [sparsity(sol3[k]) for k in sol3.keys()]


info.to_csv('data/slr_results/info.csv')

    