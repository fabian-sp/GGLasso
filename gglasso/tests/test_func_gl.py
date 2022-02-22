"""
@author: Fabian Schaipp
"""
import numpy as np
from numpy.testing import assert_array_almost_equal

from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL

from gglasso.solver.functional_sgl_admm import ADMM_FSGL
from gglasso.solver.ggl_helper import prox_sum_Frob, prox_od_1norm
    
def test_FSGL_SGL():
    """
    test single Graphical Lasso solver vs. funcional Graphical Lasso when M=1.
    """
    p = 20
    M = 1
    N = 100

    Sigma, Theta = generate_precision_matrix(p=p, M=2, style = 'erdos', gamma = 2.8,\
                                             prob = 0.1, scale = False, seed = 123)
    S, samples = sample_covariance_matrix(Sigma, N, seed = 123)    
    lambda1 = 0.01

    Omega_0 = np.eye(p)
    
    sol1, info1 = ADMM_SGL(S, lambda1, Omega_0, tol=1e-10, rtol=1e-10, update_rho=True,\
                           verbose=True, latent=False)
    
    sol2, info2 = ADMM_FSGL(S, lambda1, M, Omega_0, tol=1e-10, rtol=1e-10,\
                            update_rho=True, verbose=True, latent=False, mu1=None)
        
    assert_array_almost_equal(sol2['Theta'], sol1['Theta'], 5)
    
    return


def test_prox_Frob():
    """ for M=1, the two prox operators are identical
    """
    p = 50
    X = np.random.randn(p,p)
    X = X+X.T

    Y1 = prox_sum_Frob(X, M=1, l=0.01)
    Y2 = prox_od_1norm(X, l=0.01)

    assert_array_almost_equal(Y1,Y2, 5)

    return