"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns

from microbiome_helper import load_and_transform
from gglasso.solver.ext_admm_solver import ext_ADMM_MGL

from gglasso.helper.experiment_helper import lambda_parametrizer
from gglasso.helper.ext_admm_helper import get_K_identity, check_G, load_G, save_G
from gglasso.helper.model_selection import model_select

K = 5
reg = 'GGL'

all_csv, S, G, ix_location, ix_exist, p, num_samples = load_and_transform(K, min_inst = 3, compute_G = False)



#save_G('data/slr_data/', G)
G = load_G('data/slr_data/')
check_G(G, p)

#%%

AIC, BIC, L1, L2, ix, SP, SKIP = model_select(ext_ADMM_MGL, S, num_samples, p, reg, method = 'BIC', G = G, gridsize1 = 3, gridsize2 = 2)

#%%
Omega_0 = get_K_identity(p)


lambda2 = 0.1
lambda1 = lambda_parametrizer(lambda2, 0.5)
sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True)

Theta = sol['Theta']

