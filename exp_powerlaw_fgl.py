"""
author: Fabian Schaipp

Sigma denotes the covariance matrix, Theta the precision matrix
"""
from time import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.covariance import GraphicalLasso

from gglasso.solver.ggl_solver import PPDNA, warmPPDNA
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, aic, error
from gglasso.helper.experiment_helper import draw_group_heatmap, plot_evolution, plot_deviation, get_default_color_coding, multiple_heatmap_animation

#from tvgl3.TVGL3 import TVGLwrapper
from regain.covariance import LatentTimeGraphicalLasso

p = 100
K = 10
N = 2000
M = 5
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
#from inverse_covariance import QuicGraphicalLasso

#quic = QuicGraphicalLasso(lam = .2, tol = 1e-6)
singleGL = GraphicalLasso(alpha = 1.5*lambda1, tol = 1e-6, max_iter = 200, verbose = True)

res = np.zeros((K,p,p))

for k in np.arange(K):
    #model = quic.fit(S[k,:,:], verbose = 1)
    model = singleGL.fit(sample[k,:,:].T)
    
    res[k,:,:] = model.precision_


results['GLASSO'] = {'Theta' : res}

#%%
# solve with PPDNA
Omega_0 = results.get('GLASSO').get('Theta')
Theta_0 = Omega_0.copy()
X_0 = np.zeros((K,p,p))

start = time()
#sol, info = PPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, eps_ppdna = 1e-3, verbose = True)

sol, info = warmPPDNA(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, admm_params = None, ppdna_params = None, eps = 1e-4 , verbose = True, measure = True)
end = time()

print(f"Running time for PPDNA was {end-start} seconds")
results['PPDNA'] = {'Theta' : sol['Theta']}

#%%
# solve with general ADMM
start = time()
sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, Theta_0 = Theta_0, X_0 = X_0, rho = 1, max_iter = 100, \
                                                        eps_admm = 1e-4, verbose = True, measure = True)
end = time()

print(f"Running time for ADMM was {end-start} seconds")

results['ADMM'] = {'Theta' : sol['Theta']}

#print(np.linalg.norm(results.get('ADMM').get('Theta') - results.get('PPDNA').get('Theta')))


#%%
# solve with TVGL

#start = time()
#thetSet = TVGLwrapper(sample, lambda1, lambda2)
#end = time()

#%%
start = time()
ltgl = LatentTimeGraphicalLasso(alpha = N*lambda1, beta = N*lambda2, tau = N*0.1, psi = 'l1', rho = 1, tol = 1e-4, max_iter=2000, verbose = True)
ltgl = ltgl.fit(sample.transpose(0,2,1))
end = time()

print(f"Running time for LGTL was {end-start}  seconds")

#results['LGTL'] = {'Theta' : ltgl.precision_}

tmp1 = ltgl.precision_
tmp2 = ltgl.latent_
draw_group_heatmap(ltgl.precision_)


#%%
Theta_admm = results.get('ADMM').get('Theta')
Theta_ppdna = results.get('PPDNA').get('Theta')
Theta_glasso = results.get('GLASSO').get('Theta')


plot_evolution(results, block = 0, L = L)

plot_deviation(results)

multiple_heatmap_animation(Theta, results, save = False)



