"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from microbiome_helper import load_and_transform, load_tax_data, all_assort_coeff, consensus
from gglasso.solver.ext_admm_solver import ext_ADMM_MGL
from gglasso.solver.single_admm_solver import ADMM_SGL

from gglasso.helper.experiment_helper import sparsity
from gglasso.helper.ext_admm_helper import get_K_identity, check_G, load_G, save_G
from gglasso.helper.model_selection import grid_search, single_range_search, ebic, surface_plot, map_l_to_w

K = 26
reg = 'GGL'

all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 5, compute_G = False)

#save_G('data/slr_results/', G)
G = load_G('data/slr_results/')
check_G(G, p)

#%%


l1 = np.linspace(1, 0.4, 2)
l1 = np.append(l1, np.linspace(0.2, 0.05, 7))
#l1 = np.linspace(0.135, 0.12, 5)
#w2 = np.logspace(-1, -5, 4)
w2 = np.linspace(0.02, 0.01, 5)

AIC, BIC, L1, L2, ix, SP, SKIP, sol1 = grid_search(ext_ADMM_MGL, S, num_samples, p, reg, l1 = l1, method = 'eBIC', w2 = w2, G = G)

W1 = map_l_to_w(L1,L2)[0]
W2 = map_l_to_w(L1,L2)[1]

#surface_plot(L1,L2, BIC, save = False)
surface_plot(W1,W2, BIC, save = True)


sol1 = sol1['Theta']

#%%

#l1 = np.linspace(0.2, 0.05, 5)
#l1 = 5*np.logspace(-1, -2.5, 6)

sAIC, sBIC, sSP, sol2, sol3, ix_uniform = single_range_search(S, l1, num_samples)

#%%

Omega_0 = get_K_identity(p)
lambda1 = 0.125
lambda2 = 0.04419 # w2 = 0.2
sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True)

res_multiple2 = sol['Theta']


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

save_result(sol1, 'multiple')

for j in np.arange(BIC.shape[0]):
    np.savetxt('data/slr_results/res_multiple/BIC_' + str(j) + '.csv', BIC[j,:,:])
np.savetxt('data/slr_results/res_multiple/AIC.csv', AIC)
np.savetxt('data/slr_results/res_multiple/SP.csv', SP)
np.savetxt('data/slr_results/res_multiple/L1.csv', L1)
np.savetxt('data/slr_results/res_multiple/L2.csv', L2)


#save_result(res_multiple2, 'multiple2')

#%%
save_result(sol2, 'single_unif')
save_result(sol3, 'single')

for j in np.arange(sBIC.shape[0]):
    np.savetxt('data/slr_results/res_single/BIC_' + str(j) + '.csv', sBIC[j,:,:])
np.savetxt('data/slr_results/res_single/AIC.csv', sAIC)
np.savetxt('data/slr_results/res_single/SP.csv', sSP)


#%%
# section for loading results
AIC = np.loadtxt('data/slr_results/res_multiple/AIC.csv')
SP = np.loadtxt('data/slr_results/res_multiple/SP.csv')
L1 = np.loadtxt('data/slr_results/res_multiple/L1.csv')
L2 = np.loadtxt('data/slr_results/res_multiple/L2.csv')

BIC = np.zeros((4, L1.shape[0], L1.shape[1]))
for j in np.arange(4):
    BIC[j,:,:] = np.loadtxt('data/slr_results/res_multiple/BIC_' + str(j) + '.csv')

sol1 = dict()  
for k in np.arange(K):
    sol1[k] = pd.read_csv('data/slr_results/res_multiple/theta_' + str(k+1) + '.csv', index_col = 0).values
    #sol1[k] = pd.read_csv('data/slr_results/res_multiple2/theta_' + str(k+1) + '.csv', index_col = 0).values
    
sol2 = dict()  
for k in np.arange(K):
    sol2[k] = pd.read_csv('data/slr_results/res_single_unif/theta_' + str(k+1) + '.csv', index_col = 0).values
    
sol3 = dict()  
for k in np.arange(K):
    sol3[k] = pd.read_csv('data/slr_results/res_single/theta_' + str(k+1) + '.csv', index_col = 0).values
#%%
########## EVALUATION ########################    
  
info = pd.DataFrame(index = np.arange(K)+1)
info['samples'] = num_samples
info['OTUs'] = p
info['group entry ratio'] = np.round((G[1,:,:] != -1).sum(axis=0) / (p*(p-1)/2),4)

info['sparsity GGL'] = [np.round(sparsity(sol1[k]), 4) for k in sol1.keys()]
info['sparsity s/u'] = [np.round(sparsity(sol2[k]), 4) for k in sol2.keys()]
info['sparsity s/i'] = [np.round(sparsity(sol3[k]), 4) for k in sol3.keys()]


info.to_csv('data/slr_results/info.csv')

#%% 
nnz1 = consensus(sol1,G)
nnz2 = consensus(sol2,G)
nnz3 = consensus(sol3,G)

consensus_min = 6

(nnz1 >=  consensus_min).sum()
(nnz2 >= consensus_min).sum()
(nnz3 >= consensus_min).sum()

plt.figure()
sns.kdeplot(nnz1, shade = True, gridsize = 10)
sns.kdeplot(nnz2, shade = True, gridsize = 10)
sns.kdeplot(nnz3, shade = True, gridsize = 10)

  
plt.hist(nnz1, bins = np.arange(26), alpha = 0.5, width = 1, log = True)
plt.hist(nnz2, bins = np.arange(26), alpha = 0.5, width = 1, log = True)
plt.hist(nnz3, bins = np.arange(26), alpha = 0.5, width = 1, log = True)
#%% 
all_tax = load_tax_data(K)


df_assort = pd.DataFrame(index = np.arange(K))

df_assort['GGL'] = all_assort_coeff(sol1, all_csv, all_tax)
df_assort['single/uniform'] = all_assort_coeff(sol2, all_csv, all_tax)
df_assort['single/indv'] = all_assort_coeff(sol3, all_csv, all_tax)

df_assort.to_csv('data/slr_results/assort.csv')