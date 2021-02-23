import pytest as pt
import numpy as np
from sklearn.covariance import GraphicalLasso

from gglasso.helper.data_generation import group_power_network, time_varying_power_network,  sample_covariance_matrix
from gglasso.helper.experiment_helper import get_K_identity
from gglasso.solver.single_admm_solver import ADMM_SGL, block_SGL
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.solver.ppdna_solver import PPDNA, warmPPDNA

from gglasso.solver.ext_admm_solver import ext_ADMM_MGL
from gglasso.helper.ext_admm_helper import construct_trivial_G


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
    
    sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, eps_admm = 1e-5, verbose = False, latent = latent, mu1 = 0.01)
      
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
    
    solext, _ = ext_ADMM_MGL(Sdict, lambda1, lambda2/np.sqrt(K), 'GGL', Omega_0, G, eps_admm = 1e-5, verbose = False, latent = latent, mu1 = 0.01)
    
    Omega_0_arr = get_K_identity(K,p)
    solADMM, _ = ADMM_MGL(S, lambda1, lambda2, 'GGL', Omega_0_arr, eps_admm = 1e-5, verbose = False, latent = latent, mu1 = 0.01)
    
    for k in np.arange(K):
        assert solext['Theta'][k] == pt.approx(solADMM['Theta'][k,:,:], abs = 1e-2), f"Absolute error in norm: {np.linalg.norm(solext['Theta'][k] - solADMM['Theta'][k,:,:])}"
    
    if latent:
        for k in np.arange(K):
            assert solext['L'][k] == pt.approx(solADMM['L'][k,:,:], abs = 1e-2), f"Absolute error in norm: {np.linalg.norm(solext['L'][k] - solADMM['L'][k,:,:])}"
    
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
    K = 1
    N = 1000
    M = 2
    lambda1 = 0.1
    
    Sigma, Theta = group_power_network(p, K, M)
    S, samples = sample_covariance_matrix(Sigma, N)    
    S = S.squeeze()
    
    Omega_0 = np.eye(p)
    
    full_sol,_ = ADMM_SGL(S, lambda1, Omega_0, eps_admm = 1e-7, verbose = False)
    
    block_sol = block_SGL(S, lambda1, Omega_0, tol = 1e-7, verbose = False)
    
    sol1 = full_sol['Theta']
    sol2 = block_sol['Theta']
    
    assert sol1 == pt.approx(sol2, abs = 1e-3), f"Absolute error in norm: {np.linalg.norm(sol1-sol2)}"
    
    return

def template_admm_vs_ppdna(p = 100, K = 5, N = 1000, reg = "GGL"):
    M = 10 # M should be divisor of p

    if reg == 'GGL':
        Sigma, Theta = group_power_network(p, K, M)
    elif reg == 'FGL':
        Sigma, Theta = time_varying_power_network(p, K, M)
    
    S, samples = sample_covariance_matrix(Sigma, N)

    lambda1= 0.05
    lambda2 = 0.01
    
    Omega_0 = get_K_identity(K,p)
    
    sol, info = ADMM_MGL(S, lambda1, lambda2, reg, Omega_0, eps_admm = 1e-6, verbose = False, latent = False)
    sol2, info2 = warmPPDNA(S, lambda1, lambda2, reg, Omega_0, eps = 1e-6 , verbose = False, measure = False)
    
    
    assert sol['Theta'] == pt.approx(sol2['Theta'], abs = 1e-2), f"Absolute error in norm: {np.linalg.norm(sol['Theta']-sol2['Theta'])}"
    
    return 
    
def test_admm_ppdna_ggl():
    template_admm_vs_ppdna(p = 100, K = 5, N = 2000, reg = "GGL")
    return

def test_admm_ppdna_fgl():
    template_admm_vs_ppdna(p = 100, K = 5, N = 2000, reg = "FGL")
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

    Sigma, Theta = group_power_network(p, K = 1, M = 2)

    S, samples = sample_covariance_matrix(Sigma, N)  # sample from multivar_norm(Sigma)
    
    # only have 1 instance, i.e. K = 1
    S = S[0,:,:]  
    Theta = Theta[0,:,:]
    samples = samples[0,:,:]

    lambda1 = 0.01

    singleGL = GraphicalLasso(alpha=lambda1, tol=1e-6, max_iter=500, verbose=False)
    model = singleGL.fit(samples.T)  # transpose because of sklearn format

    sol_scikit = model.precision_

    Omega_0 = np.eye(p)
    sol, info = ADMM_SGL(S, lambda1, Omega_0, eps_admm=1e-6, verbose=False, latent=False)

    assert sol_scikit == pt.approx(sol['Theta'], rel = 1e-3), f"Absolute error in norm: {np.linalg.norm(sol['Theta']-sol_scikit)}"

    return



