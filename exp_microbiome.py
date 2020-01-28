"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns

from microbiome_helper import load_and_transform
from gglasso.solver.ext_admm_solver import ext_ADMM_MGL

from gglasso.helper.experiment_helper import surface_plot
from gglasso.helper.ext_admm_helper import get_K_identity, check_G, load_G, save_G
from gglasso.helper.model_selection import model_select, ebic

K = 26
reg = 'GGL'

all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 5, compute_G = False)



#save_G('data/slr_data/', G)
G = load_G('data/slr_data/')
check_G(G, p)

#%%

AIC, BIC, L1, L2, ix, SP, SKIP, sol = model_select(ext_ADMM_MGL, S, num_samples, p, reg, method = 'BIC', G = G, gridsize1 = 5, gridsize2 = 4)

surface_plot(L1,L2, BIC)


#%%
L = G.shape[1]

groupsize = G[G!=-1].sum(axis=2)

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


lambda2 = 0.0116
lambda1 = 0.0736
sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True)

Theta = sol['Theta']


#%%
for k in np.arange(K):
    res_k = pd.DataFrame(sol['Theta'][k], index = all_csv[k].index, columns = all_csv[k].index)
    res_k.to_csv('data/slr_results/theta_' + str(k+1) + ".csv")
