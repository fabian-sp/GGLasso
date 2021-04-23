"""
Basic example
=================================

We demonstrate how to use ``GGLasso`` for a SGL problem.

"""

from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix
from gglasso.problem import glasso_problem
from gglasso.helper.basic_linalg import adjacency_matrix

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

p = 10
N = 20

Sigma, Theta = group_power_network(p, K = 1, M = 1)
S, sample = sample_covariance_matrix(Sigma, N)
# remove redundant dimension
Theta = Theta[0]; S = S[0]; sample= sample[0]

print("Shape of empirical covariance matrix: ", S.shape)
print("Shape of the sample array: ", sample.shape)


# Draw the graph of the true precision matrix.

A = adjacency_matrix(Theta)
np.fill_diagonal(A,1)

G = nx.from_numpy_array(A)
nx.draw_networkx(G)


#%%
# We now create an instance of ``glasso_problem``. The problem formulation is derived automatically from the input shape of ``S``.

P = glasso_problem(S, N, reg_params = None, latent = False)

print(P)

