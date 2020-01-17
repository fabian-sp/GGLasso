"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd

from microbiome_helper import load_and_transform
from gglasso.solver.ext_admm_solver import ext_ADMM_MGL

from gglasso.helper.experiment_helper import lambda_parametrizer
from gglasso.helper.ext_admm_helper import get_K_identity, check_G, load_G, save_G


K = 5

lambda2 = 0.1
lambda1 = lambda_parametrizer(lambda2, 0.5)

all_csv, S, G, ix_location = load_and_transform(K, min_inst = 3, compute_G = True)


#save_G('data/slr_data/', G)
#load_G('data/slr_data/')

p = np.zeros(K, dtype= int)
for k in np.arange(K):
    p[k] = S[k].shape[0]
    
check_G(G, p)

Omega_0 = get_K_identity(p)

sol, info = ext_ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True)


#from gglasso.solver.ext_latent_admm_solver import ext_ADMM_MGL
#mu1 = .1
#sol, info = ext_ADMM_MGL(S, lambda1, lambda2, mu1, 'GGL', Omega_0, G, eps_admm = 1e-3, verbose = True)


Theta = sol['Theta']
