import pytest
import numpy as np
from sklearn.covariance import GraphicalLasso
from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL



def test_similarity():
    p = 10
    N = 100

    Sigma, Theta = group_power_network(p, K=5, M=2)

    S, samples = sample_covariance_matrix(Sigma, N)  # sample from multivar_norm(Sigma)

    S = S[0, :, :]  # take the first block K = 1
    Theta = Theta[0, :, :]

    lambda1 = 0.01

    singleGL = GraphicalLasso(alpha=lambda1, tol=1e-6, max_iter=500, verbose=True)
    model = singleGL.fit(samples[0, :, :].T)  # transpose because of sklearn format

    res_scikit = model.precision_

    Omega_0 = np.eye(p)
    sol, info = ADMM_SGL(S, lambda1, Omega_0, eps_admm=1e-4, verbose=True, latent=False)

    assert res_scikit == pytest.approx(sol['Theta'], rel=1e-2)