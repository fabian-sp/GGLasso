from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
from numpy.linalg import norm

from gglasso.Lvsglasso.lvsglasso.proximal import block_thresh, block_thresh_2d, block_thresh_entrywise, prox_nuc
from gglasso.Lvsglasso.lvsglasso.util import pack_and_stack, partial_coher
from gglasso.Lvsglasso.lvsglasso.lvsglasso import admm_consensus

from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL

def test_latent_variables():
    p = 10
    N = 100

    Sigma, Theta = group_power_network(p, K=5, M=2)

    S, samples = sample_covariance_matrix(Sigma, N)  # sample from multivar_norm(Sigma)

    S = S[0, :, :]  # take the first block K = 1
    Theta = Theta[0, :, :]

    lambda1 = 0.01

    Omega_0 = np.eye(p)
    mu1 = .01

    sol, info = ADMM_SGL(S, lambda1, Omega_0, eps_admm=1e-4, verbose=True, latent=True, mu1=mu1)


    S_, L_, obj_inds, infeas_inds_, diagnostics, lamS, lamL = admm_consensus(S[np.newaxis,:,:], lambda1, mu1, niter=100, alpha=1., mu=None,
                       mu_cont=None, mu_cont_iter=10, mu_min=1e-6,
                       S_init=None, L_init=None,
                       abstol=1e-5, reltol=1e-5, verbose=False, compute_obj=False,
                       compute_infeas=False, do_lowrank=True, do_sparse=True)

    new_sol = S_[:1,:,:].reshape(10,10)

    assert new_sol == pytest.approx(sol['Theta'], rel=1e-1)


# p = 10
# N = 100
#
# Sigma, Theta = group_power_network(p, K=5, M=2)
#
# S, samples = sample_covariance_matrix(Sigma, N)  # sample from multivar_norm(Sigma)
#
# S = S[0, :, :]  # take the first block K = 1
# Theta = Theta[0, :, :]
#
# lambda1 = 0.01
#
# Omega_0 = np.eye(p)
# mu1 = .01
#
# sol, info = ADMM_SGL(S, lambda1, Omega_0, eps_admm=1e-4, verbose=True, latent=True, mu1=mu1)
#
#
# S_, L_, obj_inds, infeas_inds_, diagnostics, lamS, lamL = admm_consensus(S[np.newaxis,:,:], lambda1, mu1, niter=100, alpha=1., mu=None,
#                    mu_cont=None, mu_cont_iter=10, mu_min=1e-6,
#                    S_init=None, L_init=None,
#                    abstol=1e-5, reltol=1e-5, verbose=False, compute_obj=False,
#                    compute_infeas=False, do_lowrank=True, do_sparse=True)
#
# new_sol = S_[:1,:,:].reshape(10,10)
#
# print(norm(sol['Theta'] - new_sol))
# The difference is 0.98658