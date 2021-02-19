import numpy as np
import time

from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.problem import glasso_problem

p = 100
K = 5
N = 100
M = 2

reg = 'GGL'

if reg == 'GGL':
    Sigma, Theta = group_power_network(p, K, M)
elif reg == 'FGL':
    Sigma, Theta = time_varying_power_network(p, K, M)

S, samples = sample_covariance_matrix(Sigma, N)


P = glasso_problem(S = S, N = N, reg = reg, latent = False)

print(P)

reg_params = {'lambda1': 0.05, 'lambda2' : 0.01}


P.set_reg_params(reg_params)

solver_params = {'verbose': True, 'measure': False}

start = time.time()
P.solve(solver_params= solver_params)
end = time.time(); print(end-start)

Theta_sol = P.solution.precision_
A_sol = P.solution.adjacency_



#%% scale test

S2 = 2*S

P2 = glasso_problem(S = S2, N = N, reg = reg, latent = False)

P2._scale_input_to_correlation()

P2.set_reg_params(reg_params)

start = time.time()
P2.solve(solver_params= solver_params)
end = time.time(); print(end-start)


#%% non-conforming example












#%% SGL example
S0 = S[0,:,:].copy()

P = glasso_problem(S = S0, N = N, reg = None, latent = False)


reg_params = {'lambda1': 0.1}
P.set_reg_params(reg_params)

start = time.time()
P.solve()
end = time.time(); print(end-start)

Theta_sol = P.solution.precision_
A_sol = P.solution.adjacency_

