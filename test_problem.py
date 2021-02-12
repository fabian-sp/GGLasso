import numpy as np
import time

from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.problem import glasso_problem

p = 100
K = 5
N = 1000
M = 2

reg = 'GGL'

if reg == 'GGL':
    Sigma, Theta = group_power_network(p, K, M)
elif reg == 'FGL':
    Sigma, Theta = time_varying_power_network(p, K, M)

S, samples = sample_covariance_matrix(Sigma, N)



P = glasso_problem(S = S, N = N, reg = reg, latent = False)

print(P)

reg_params = {'lambda1': 0.05, 'lambda2' : 0.05}


P.set_reg_params(reg_params)

solver_params = {'verbose': True, 'measure': False}

start = time.time()
P.solve(solver_params= solver_params)
end = time.time(); print(end-start)

