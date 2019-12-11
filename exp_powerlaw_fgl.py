"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""
from time import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from inverse_covariance import QuicGraphicalLasso

from gglasso.solver.ggl_solver import PPDNA, warmPPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, aic, error, draw_group_heatmap, plot_evolution, get_default_color_coding

from tvgl3.TVGL3 import TVGLwrapper

p = 100
K = 10
N = 2000
M = 10
L = int(p/M)

reg = 'FGL'

Sigma, Theta = time_varying_power_network(p, K, M)
#np.linalg.norm(np.eye(p) - Sigma@Theta)/np.linalg.norm(np.eye(p))

draw_group_heatmap(Theta)

S,sample = sample_covariance_matrix(Sigma, N)
Sinv = np.linalg.pinv(S, hermitian = True)



lambda1 = 0.05
lambda2 = 0.1

methods = ['PPDNA', 'ADMM', 'GLASSO']
color_dict = get_default_color_coding()

results = {}
results['truth'] = {'Theta' : Theta}

#%%
# solve with QUIC/single Glasso
quic = QuicGraphicalLasso(lam = .2, tol = 1e-6)
res = np.zeros((K,p,p))

for k in np.arange(K):
    model = quic.fit(S[k,:,:], verbose = 1)
    res[k,:,:] = model.precision_

#res = prox_p2(res, .01)

results['GLASSO'] = {'Theta' : res}

#%%
# solve with PPDNA
Omega_0 = results.get('GLASSO').get('Theta')
Theta_0 = Omega_0.copy()
X_0 = np.zeros((K,p,p))

start = time()
#sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, eps_ppdna = 1e-3, verbose = True)

sol, info = warmPPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, admm_params = None, ppdna_params = None, eps = 1e-5 , verbose = True, measure = True)
end = time()

print(f"Running time for PPDNA was {end-start} seconds")
results['PPDNA'] = {'Theta' : sol['Theta']}

#%%
# solve with general ADMM
start = time()
sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, rho = 1, max_iter = 100, \
                                                        eps_admm = 1e-5, verbose = True, measure = True)
end = time()

print(f"Running time for ADMM was {end-start} seconds")

results['ADMM'] = {'Theta' : sol['Theta']}

#print(np.linalg.norm(results.get('ADMM').get('Theta') - results.get('PPDNA').get('Theta')))


#%%
# solve with TVGL

start = time()
thetSet = TVGLwrapper(sample, lambda1, lambda2)
end = time()

#%%
Theta_admm = results.get('ADMM').get('Theta')
Theta_ppdna = results.get('PPDNA').get('Theta')
Theta_glasso = results.get('GLASSO').get('Theta')


plot_evolution(results, block = 0, L = L)


def l1norm_od(Theta):
    """
    calculates the off-diagonal l1-norm of a matrix
    """
    (p1,p2) = Theta.shape
    res = 0
    for i in np.arange(p1):
        for j in np.arange(p2):
            if i == j:
                continue
            else:
                res += abs(Theta[i,j])
                
    return res

def deviation(Theta):
    """
    calculates the deviation of subsequent Theta estimates
    deviation = off-diagonal l1 norm
    """
    #tmp = np.roll(Theta, 1, axis = 0)
    (K,p,p) = Theta.shape
    d = np.zeros(K-1)
    for k in np.arange(K-1):
        d[k] = l1norm_od(Theta[k+1,:,:] - Theta[k,:,:])
        
    return d

#%%
plot_aesthetics = {'marker' : 'o', 'linestyle' : '-', 'markersize' : 5}
plt.figure()

for m in list(results.keys()):
    d = deviation(results.get(m).get('Theta'))
    with sns.axes_style("whitegrid"):
        plt.plot(d, c = color_dict[m], **plot_aesthetics)
        
plt.ylabel('Temporal Deviation')
plt.xlabel('Time (k=1,...,K)')
plt.legend(labels = list(results.keys()))


