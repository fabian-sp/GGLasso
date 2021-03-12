import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.helper.model_selection import single_grid_search

from gglasso.helper.basic_linalg import scale_array_by_diagonal

p = 100
N = 10000

# generate Theta
A = np.random.randn(p,p)
Theta = A.T@A + 150*np.eye(p)

scale = np.tile(np.sqrt(np.diag(Theta)),(p,1))
scale = scale.T * scale
        
Theta = Theta/scale
Theta[np.abs(Theta) <= 0.05] = 0

assert np.all(np.linalg.eigvalsh(Theta) > 0)
print("smallest eigval of Theta:", np.linalg.eigvalsh(Theta).min())

# generate L
L = np.zeros((p,p))
rank = 5
for i in range(rank):
    v = np.random.rand(p)
    L += np.outer(v,v)

L = L *(0.6/np.linalg.eigvalsh(L).max()) 
print("Rank of L:", np.linalg.matrix_rank(L))


R = Theta - L
print("smallest eigval of R:", np.linalg.eigvalsh(R).min())
assert np.all(np.linalg.eigvalsh(R) > 0)

# generate S
Sigma = np.linalg.pinv(R)

sample = np.zeros((p,N))
sample = np.random.multivariate_normal(np.zeros(p), Sigma, N).T

S = np.cov(sample, bias = True)

S = scale_array_by_diagonal(S)

D = np.linalg.eigvalsh(S)
print("Eigvals of S:", np.linalg.eigvalsh(S))



#%%
Omega_0 = np.eye(p)
lambda1 = 0.1

sol, info = ADMM_SGL(S, lambda1, Omega_0 , tol = 1e-5 , rtol = 1e-4, verbose = False, latent = True, mu1 = 1/D.min())
np.linalg.matrix_rank(sol['L'])

M=25
mu_range = np.logspace(-2,1,M) 
#mu_range = np.linspace(1/D.max(), 1/D.min(), 20)

lambda_range = np.logspace(-3,1,10)


best_sol, estimates, lowrank, stats = single_grid_search(S, lambda_range, N, method = 'eBIC', gamma = 0.1, latent = True, mu_range = mu_range)

#%%
fig, axs = plt.subplots(1,2)

sns.heatmap(stats["RANK"], annot = True, xticklabels = np.round(mu_range,2), yticklabels = np.round(lambda_range,2), cbar = False, ax = axs[0])
axs[0].set_title("Rankf of L")
sns.heatmap(np.round(stats["SP"],2), annot = True, xticklabels = np.round(mu_range,2), yticklabels = np.round(lambda_range,2), cbar = False, cmap = "coolwarm", ax = axs[1])
axs[1].set_title("Sparsity of Theta")


