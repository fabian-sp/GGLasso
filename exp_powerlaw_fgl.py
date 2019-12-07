"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""
from time import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from inverse_covariance import QuicGraphicalLasso

from gglasso.solver.ggl_solver import PPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, aic, error, draw_group_graph, draw_group_heatmap, default_color_coding


p = 100
K = 10
N = 2000
M = 10
L = int(p/M)

reg = 'FGL'

Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)/np.linalg.norm(np.eye(p))

draw_group_heatmap(Theta)

S = sample_covariance_matrix(Sigma, N)
Sinv = np.linalg.pinv(S, hermitian = True)


#%%
lambda1 = 0.1
lambda2 = 0.1

methods = ['PPDNA', 'ADMM', 'GLASSO']
color_dict = default_color_coding()

results = {}

#%%
# solve with QUIC/single Glasso
quic = QuicGraphicalLasso(lam = lambda1, tol = 1e-6)
res = np.zeros((K,p,p))

for k in np.arange(K):
    model = quic.fit(S[k,:,:], verbose = 1)
    res[k,:,:] = model.precision_

results['GLASSO'] = {'Theta' : res}

#%%
Omega_0 = results.get('GLASSO').get('Theta')
Theta_0 = Omega_0.copy()
X_0 = np.zeros((K,p,p))

start = time()
sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, sigma_0 = 10, max_iter = 20, \
                                                        eps_ppdna = 1e-3, verbose = True)
end = time()

print(f"Running time for PPDNA was {end-start} seconds")
results['PPDNA'] = {'Theta' : sol['Theta']}

#%%
start = time()
sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, rho = 1, max_iter = 100, \
                                                        eps_admm = 1e-3, verbose = True)
end = time()

print(f"Running time for ADMM was {end-start} seconds")

results['ADMM'] = {'Theta' : sol['Theta']}

#print(np.linalg.norm(results.get('ADMM').get('Theta') - results.get('PPDNA').get('Theta')))

Theta_admm = results.get('ADMM').get('Theta')
Theta_ppdna = results.get('PPDNA').get('Theta')
Theta_glasso = results.get('GLASSO').get('Theta')
#%%

plt.figure()

for i in np.arange(L):
    for j in np.arange(start = i+1, stop = L):
        
        x = np.arange(K)
        plt.plot(x, abs(Theta[:,i,j]), c=color_dict['truth'], label = 'True Network' if (i == 0) & (j == 1) else "")
        
        for m in methods:
            T = results.get(m).get('Theta')
            plt.plot(x, abs(T[:,i,j]), c=color_dict[m], label = m if (i == 0) & (j == 1) else "")
            
plt.legend()        



#%%
method = 'ADMM'

fig,axs = plt.subplots(nrows = 1, ncols = 2)
draw_group_heatmap(Theta, axs[0])
draw_group_heatmap(results.get(method).get('Theta'), axs[1])
