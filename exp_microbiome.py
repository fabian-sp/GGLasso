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


from gglasso.helper.ext_admm_helper import get_K_identity, check_G, load_G, save_G
from gglasso.helper.model_selection import model_select, ebic, surface_plot

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

AIC, BIC, L1, L2, ix, SP, SKIP, sol = model_select(ext_ADMM_MGL, S, num_samples, p, reg, method = 'BIC', l1 = l1, w2 = w2, G = G)

res_multiple = sol['Theta']
surface_plot(L1,L2, BIC, save = False)

#%%
L = G.shape[1]
groupsize = (G!=-1).sum(axis=2)[0]

nnz = np.zeros(L)
Theta = sol['Theta']
for l in np.arange(L):
    for k in np.arange(K):
        if G[0,l,k] == -1:
            continue
        else:
            nnz[l] += abs(Theta[k][G[0,l,k], G[1,l,k]]) >= 1e-5
    
    
#%%

Omega_0 = get_K_identity(p)
lambda1 = 0.0875
lambda2 = 0.00003
sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True)

res_multiple = sol['Theta']

#%%
# compute single GL solution

res_single = dict()
for k in np.arange(K):
    Omega_0 = np.eye(p[k])
    sol1, info1 = ADMM_SGL(S[k], lambda1, Omega_0, eps_admm = 1e-3, verbose = True)
    
    res_single[k] = sol1['Theta']

#%%
# section for saving results

info = pd.DataFrame(index = np.arange(K)+1)
info['samples'] = num_samples
info['OTUs'] = p
info['off-diagonals'] = (p*(p-1)/2).astype(int)
info['group entries'] = (G[1,:,:] != -1).sum(axis=0)

info.to_csv('data/slr_results/info.csv')

def save_result(res, typ):
    path = 'data/slr_results/res_' + typ
    K = len(res.keys())
    for k in np.arange(K):
        res_k = pd.DataFrame(res[k], index = all_csv[k].index, columns = all_csv[k].index)
        res_k.to_csv(path + '/theta_' + str(k+1) + ".csv")
    print("All files saved")
    return

save_result(res_multiple, 'multiple')
save_result(res_single, 'single')

for j in np.arange(BIC.shape[0]):
    np.savetxt('data/slr_results/BIC_' + str(j) + '.csv', BIC[j,:,:])
np.savetxt('data/slr_results/AIC.csv', AIC)
np.savetxt('data/slr_results/SP.csv', SP)
np.savetxt('data/slr_results/L1.csv', L1)
np.savetxt('data/slr_results/L2.csv', L2)

#%%
# section for loading results
AIC = np.loadtxt('data/slr_results/AIC.csv')
SP = np.loadtxt('data/slr_results/SP.csv')
L1 = np.loadtxt('data/slr_results/L1.csv')
L2 = np.loadtxt('data/slr_results/L2.csv')

BIC = np.zeros((4, L1.shape[0], L1.shape[1]))
for j in np.arange(4):
    BIC[j,:,:] = np.loadtxt('data/slr_results/BIC_' + str(j) + '.csv')