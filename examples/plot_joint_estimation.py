"""
Joint estimation vs. SGL
=================================

We compare runtimes for solving K SGL problems vs. solving a joint Graphical Lasso problem with K instances.

"""

from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.single_admm_solver import block_SGL, ADMM_SGL
from gglasso.helper.utils import get_K_identity

import numpy as np
import matplotlib.pyplot as plt
import time


p = 50
N = 50

lambda1 = 0.1
lambda2 = 0.05


allK = [2, 5, 10, 15, 25]
allS = list()

J = len(allK)

rt_ggl = np.zeros(J)
rt_fgl = np.zeros(J)
rt_sgl = np.zeros(J)

for j in range(J):
    
    Sigma, Theta = group_power_network(p, K = allK[j], M = 5, scale = False, nxseed = 1235)
    S, sample = sample_covariance_matrix(Sigma, N)

    print("Shape of empirical covariance matrix: ", S.shape)
    print("Shape of the sample array: ", sample.shape)

    allS.append(S)
    
#%% initialize solvers (for numba jitting)
_,_ = ADMM_MGL(allS[0], lambda1, lambda2, "GGL", allS[0], max_iter = 10, measure = False, latent = False)
_,_ = ADMM_MGL(allS[0], lambda1, lambda2, "FGL", allS[0], max_iter = 10, measure = False, latent = False)

_ = ADMM_SGL(allS[0][0], lambda1, allS[0][0], max_iter=10, verbose=False, measure=False)

#%%
# Solve a Group Graphical Lasso problem.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#

for j in range(J):
    
    Omega_0 = get_K_identity(allK[j],p)
    start = time.time()
    sol,_ = ADMM_MGL(allS[j], lambda1, lambda2, "GGL", Omega_0, tol = 1e-7, rtol= 1e-7, measure = False, latent = False)
    end = time.time()

    rt_ggl[j] = end-start
    
#%%
# Solve a Fused Graphical Lasso problem.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#

for j in range(J):
    
    Omega_0 = get_K_identity(allK[j],p)
    start = time.time()
    sol,_ = ADMM_MGL(allS[j], lambda1, lambda2, "FGL", Omega_0, tol = 1e-7, rtol= 1e-7, measure = False, latent = False)
    end = time.time()

    rt_fgl[j] = end-start

#%%
# Solve K Single Graphical Lasso problems.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#

for j in range(J):
    
    start = time.time()
    
    Omega_0 = np.eye(p); X_0 = np.zeros(p)
    for k in range(allK[j]):
        
        #sol = block_SGL(allS[j][k,:,:], lambda1, Omega_0, tol = 1e-7, rtol= 1e-6, measure = False)
        sol,_ = ADMM_SGL(allS[j][k,:,:], lambda1, Omega_0, X_0 = X_0, tol = 1e-7, rtol= 1e-7, measure = False)
        
        # use solution as warm start
        Omega_0 = sol['Omega']
        X_0 = sol['X']
    
    end = time.time()

    rt_sgl[j] = end-start
    
    
#%%
# Plotting.
# ^^^^^^^^^^^^
#

fig, ax = plt.subplots()

ax.plot(allK, rt_ggl, '-o', label = "GGL")
ax.plot(allK, rt_fgl, '-o', label = "FGL")
ax.plot(allK, rt_sgl, '-o', label = "SGL")

ax.set_ylabel("Runtime [sec]")
ax.set_xlabel("Number of instances K")
ax.set_title(f"Runtime for p={p}, N={N}")
ax.legend()