"""
author: Fabian Schaipp
"""
import pytest as pt
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.covariance import GraphicalLasso

from gglasso.helper.data_generation import group_power_network, time_varying_power_network, generate_precision_matrix, sample_covariance_matrix
from gglasso.helper.utils import get_K_identity
from gglasso.solver.single_admm_solver import ADMM_SGL, block_SGL, get_connected_components
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.ppdna_solver import PPDNA, warmPPDNA

from gglasso.solver.ext_admm_solver import ext_ADMM_MGL
from gglasso.helper.ext_admm_helper import construct_trivial_G

from gglasso.helper.basic_linalg import scale_array_by_diagonal

###############################################################
### TEST WHETHER SOLVERS ARE RUNNING
###############################################################


def template_ADMM_MGL(p = 100, K = 5, N = 1000, reg = 'GGL', latent = False):
    """
    template for test for ADMM MGL solver with conforming variables
    """
    M = 10 # M should be divisor of p

    if reg == 'GGL':
        Sigma, Theta = group_power_network(p, K, M)
    elif reg == 'FGL':
        Sigma, Theta = time_varying_power_network(p, K, M)
    
    S, samples = sample_covariance_matrix(Sigma, N)

    lambda1= 0.05
    lambda2 = 0.01
    
    Omega_0 = get_K_identity(K,p)
    
    sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, tol = 1e-5, rtol = 1e-5, verbose = True, measure = True, latent = latent, mu1 = 0.01)
    assert info['status'] == 'optimal'
    
    _, info2 = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, tol = 1e-20, rtol = 1e-20, max_iter = 2, latent = latent, mu1 = 0.01)
    assert info2['status'] == 'max iterations reached'
    
    return

def test_ADMM_GGL(): 
    template_ADMM_MGL(p = 50, K = 3, N = 1000, reg = 'GGL', latent = False)
    return

def test_ADMM_FGL():
    template_ADMM_MGL(p = 50, K = 3, N = 1000, reg = 'FGL', latent = False)
    return

def test_ADMM_GGL_latent():
    template_ADMM_MGL(p = 50, K = 3, N = 1000, reg = 'GGL', latent = True)
    return

def test_ADMM_FGL_latent():
    template_ADMM_MGL(p = 50, K = 3, N = 1000, reg = 'FGL', latent = True)
    return

###############################################################
### TEST FOR CONSISTENCY
###############################################################

def template_extADMM_consistent(latent = False):
    """
    tests whether the extended ADMM solver results in the same as MGL-solver for the redundant case of conforming variables
    """
    p = 50
    N = 1000
    K = 5
    M = 10
    
    lambda1= 0.05
    lambda2 = 0.01
    
    Sigma, Theta = group_power_network(p, K, M)
    S, samples = sample_covariance_matrix(Sigma, N)
    
    Sdict = dict()
    Omega_0 = dict()
    
    for k in np.arange(K):
        Sdict[k] = S[k,:,:].copy()
        Omega_0[k] = np.eye(p)
    
    # constructs the "trivial" groups, i.e. all variables present in all instances  
    G = construct_trivial_G(p, K)
    
    solext, _ = ext_ADMM_MGL(Sdict, lambda1, lambda2/np.sqrt(K), 'GGL', Omega_0, G, tol = 1e-9, rtol = 1e-9, verbose = True, latent = latent, mu1 = 0.01)
    solext2, _ = ext_ADMM_MGL(Sdict, lambda1, lambda2/np.sqrt(K), 'GGL', Omega_0, G, stopping_criterion = 'kkt', tol = 1e-8, verbose = True, latent = latent, mu1 = 0.01)
    
    Omega_0_arr = get_K_identity(K,p)
    solADMM, info = ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0_arr, tol = 1e-9, rtol = 1e-9, verbose = False, latent = latent, mu1 = 0.01)
    
    
    for k in np.arange(K):
        assert_array_almost_equal(solext['Theta'][k], solADMM['Theta'][k,:,:], 2)
        assert_array_almost_equal(solext2['Theta'][k], solADMM['Theta'][k,:,:], 2)
        
    if latent:
        for k in np.arange(K):
            assert_array_almost_equal(solext['L'][k], solADMM['L'][k,:,:], 2)
            assert_array_almost_equal(solext2['L'][k], solADMM['L'][k,:,:], 2)
        
    return

def test_extADMM_consistent():
    template_extADMM_consistent(latent = False)
    return

def test_extADMM_consistent_latent():
    template_extADMM_consistent(latent = True)
    return


def test_block_SGL():
    """
    tests whether solving each connected component results in the same as solving the whole problem (see Witten et al. paper referenced in block_SGL)
    """
    p = 100
    lambda1 = 0.12
    
    np.random.seed(seed=1234)
    A = np.random.randn(p,p)
    S = A.T@A + 90*np.eye(p)
    S = scale_array_by_diagonal(S)
    
    Omega_0 = np.eye(p)
    
    full_sol,_ = ADMM_SGL(S, lambda1, Omega_0, tol = 1e-7, rtol = 1e-5, verbose = False)
    
    numC, allC =  get_connected_components(S, lambda1)
    assert numC > 1, "Test is redundant if only one connected component"
    block_sol = block_SGL(S, lambda1, Omega_0, tol = 1e-7, rtol = 1e-5, verbose = False)
    
    sol1 = full_sol['Theta']
    sol2 = block_sol['Theta']
    
    assert_array_almost_equal(sol1, sol2, 3)
    
    return

def template_admm_vs_ppdna(p = 50, K = 3, N = 1000, reg = "GGL"):
    M = 5 # M should be divisor of p

    if reg == 'GGL':
        Sigma, Theta = group_power_network(p, K, M)
    elif reg == 'FGL':
        Sigma, Theta = time_varying_power_network(p, K, M)
    
    S, samples = sample_covariance_matrix(Sigma, N)

    lambda1= 0.05
    lambda2 = 0.01
    
    Omega_0 = get_K_identity(K,p)
    
    sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, stopping_criterion = 'kkt', tol = 1e-6, rtol = 1e-5, verbose = True, latent = False)
    
    sol2, info2 = warmPPDNA(S, lambda1, lambda2, reg, Omega_0, eps = 1e-6 , verbose = False, measure = True)
    
    sol3, info3 = PPDNA(S, lambda1, lambda2, reg, Omega_0, eps_ppdna = 1e-6 , verbose = True, measure = True)
    
    
    assert_array_almost_equal(sol['Theta'], sol2['Theta'], 2)
    assert_array_almost_equal(sol2['Theta'], sol3['Theta'], 2)
    
    return 
    
def test_admm_ppdna_ggl():
    template_admm_vs_ppdna(p = 50, K = 3, N = 2000, reg = "GGL")
    return

def test_admm_ppdna_fgl():
    template_admm_vs_ppdna(p = 50, K = 3, N = 2000, reg = "FGL")
    return


###############################################################
### TEST VS. OTHER PACKAGES 
###############################################################

def test_SGL_scikit():
    """
    test single Graphical Lasso solver vs. scikit-learn
    """
    p = 10
    N = 100


    Sigma, Theta = generate_precision_matrix(p=p, M=2, style = 'erdos', gamma = 2.8, prob = 0.1, scale = False, nxseed = None)
    S, samples = sample_covariance_matrix(Sigma, N)  # sample from multivar_norm(Sigma)
    
    lambda1 = 0.01

    singleGL = GraphicalLasso(alpha=lambda1, tol=1e-6, max_iter=500, verbose=False)
    model = singleGL.fit(samples.T)  # transpose because of sklearn format

    sol_scikit = model.precision_

    Omega_0 = np.eye(p)
    
    sol, info = ADMM_SGL(S, lambda1, Omega_0, tol=1e-7, rtol=1e-5, verbose=True, latent=False)
    
    # run into max_iter
    sol2, info2 = ADMM_SGL(S, lambda1, Omega_0, stopping_criterion = 'kkt', tol=1e-20, max_iter = 200, verbose=True, latent=False)
    
    assert_array_almost_equal(sol_scikit, sol['Theta'], 3)
    assert_array_almost_equal(sol_scikit, sol2['Theta'], 3)
    
    return


