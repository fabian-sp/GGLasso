import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.linalg import block_diag
import time

from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL, block_SGL



#%%
p = 100
K = 1
N = 1000
M = 2

reg = 'GGL'


Sigma, Theta = group_power_network(p, K, M)
S, samples = sample_covariance_matrix(Sigma, N)

S = S.squeeze()


lambda1 = 0.1
Omega_0 = np.eye(p)

start = time.time()
full_sol,_ = ADMM_SGL(S, lambda1, Omega_0, eps_admm = 1e-7, verbose = True)
end = time.time()
print("Full ADMM took sec:", end-start)

start = time.time()
block_sol = block_SGL(S, lambda1, Omega_0, tol = 1e-7, verbose = False, measure = False)
end = time.time()
print("Block ADMM took sec:", end-start)

sol1 = full_sol['Theta']
sol2 = block_sol['Theta']


# check
np.linalg.norm(sol2-sol1)    

