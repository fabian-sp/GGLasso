import numpy as np
from matplotlib import pyplot as plt
import time
from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL


p = 100
N = 1000

A = np.random.rand(p,p)


Theta = A.T@A + 100*np.eye(p)

Theta[np.abs(Theta) <= 0.1] = 0

scale = np.tile(np.sqrt(np.diag(Theta)),(p,1))
scale = scale.T * scale
        
Theta = Theta/scale

assert np.all(np.linalg.eigvalsh(Theta) > 0)
print(np.linalg.eigvalsh(Theta).min())

L = np.zeros((p,p))
rank = 5
for i in range(rank):
    v = np.random.rand(p)
    L += np.outer(v,v)

print(np.linalg.matrix_rank(L))

L = L *(1/np.linalg.eigvalsh(L).max()) 

R = Theta - L
assert np.all(np.linalg.eigvalsh(R) > 0)

Sigma = np.linalg.pinv(R)


sample = np.zeros((p,N))
sample = np.random.multivariate_normal(np.zeros(p), Sigma[:,:], N).T

S = np.cov(sample, bias = True)


lambda1 = 0.1
Omega_0 = np.eye(p)

print(np.linalg.eigvalsh(S))
#%%
plt.figure()
M=20
mu_range = np.logspace(-2,0,M) 
for j in range(M):

    sol, info = ADMM_SGL(S, lambda1, Omega_0 , eps_admm = 1e-4 , verbose = False, latent = True, mu1 = mu_range[j])
    print("Rank of L: ", np.linalg.matrix_rank(sol['L']))
    
    plt.scatter(mu_range[j], np.linalg.matrix_rank(sol['L']), c = 'blue')

plt.xscale('log')






