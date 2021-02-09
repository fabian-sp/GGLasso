import numpy as np
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

lambda1= 0.05
lambda2 = 0.05




