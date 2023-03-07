"""
author: Fabian Schaipp
"""
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix, generate_precision_matrix
from gglasso.problem import glasso_problem
from gglasso.helper.ext_admm_helper import construct_trivial_G, create_group_array, construct_indexer, check_G
from gglasso.helper.basic_linalg import scale_array_by_diagonal


##########################################
# NOTE
# Numpy can not produce identical random numbers across versions/machines (even for same seed)
# Particularly the multivariate_normal function is not deterministic
# Hence, for now we do not assert on fixed outcomes.
# see: https://github.com/numpy/numpy/issues/22975
##########################################

p = 20
K = 3
N = 1000
M = 4

###############################################################
### TEST MGL
###############################################################

def template_problem_MGL(S, N, reg = 'GGL', latent = False, G = None):
    """
    template for testing the MGL problem object
    """
    P = glasso_problem(S = S, N = N, reg = reg, latent = latent, G = G)
    print(P)
    
    modelselectparams = dict()
    modelselectparams['lambda1_range'] = np.logspace(-3,0,4)
    modelselectparams['lambda2_range'] = np.logspace(-1,-3,3)
              
    if latent:
        modelselectparams['mu1_range'] = np.logspace(-2,0,4)
    else:
        modelselectparams['mu1_range'] = None
    
    
    reg_params = {'lambda1': 0.01, 'lambda2': 0.001}
    if latent:
        reg_params['mu1'] = 1.
    
    # set reg params and solve again
    P.set_reg_params(reg_params)
    P.solve(verbose = True)
    
    # test model selection
    P.model_selection(modelselect_params = modelselectparams, method = 'eBIC', gamma = 0.1)
    
    #tmp = P.modelselect_stats.copy()
    _ = P.solution.calc_ebic(gamma = 0.1)
    P.solution.calc_adjacency()
    
    return P

def test_GGL():
    Sigma, Theta = group_power_network(p, K, M, seed=123)    
    S, samples = sample_covariance_matrix(Sigma, N, seed=123)
    P = template_problem_MGL(S, N, reg = 'GGL', latent = False)   
    
    return

def test_GGL_latent():
    Sigma, Theta = group_power_network(p, K, M, seed=123)    
    S, samples = sample_covariance_matrix(Sigma, N, seed=123)
    P = template_problem_MGL(S, N, reg = 'GGL', latent = True)
    
    return

def test_FGL():
    Sigma, Theta = time_varying_power_network(p, K, M, seed=123)
    S, samples = sample_covariance_matrix(Sigma, N, seed=123)
    P = template_problem_MGL(S, N, reg = 'FGL', latent = False)
    
    return

def test_FGL_latent():
    Sigma, Theta = time_varying_power_network(p, K, M, seed=123)
    S, samples = sample_covariance_matrix(Sigma, N, seed=123)
    P = template_problem_MGL(S, N, reg = 'FGL', latent = True)
    
    return

def test_GGL_ext():
    Sigma, Theta = group_power_network(p, K, M, seed=456)
    S, samples = sample_covariance_matrix(Sigma, N, seed=456)
    
    Sdict = dict()
    for k in np.arange(K):
        Sdict[k] = S[k,:,:].copy()
        
    G = construct_trivial_G(p, K)
    P = template_problem_MGL(Sdict, N, reg = 'GGL', latent = False, G = G)
    return

def test_GGL_ext_latent():
    Sigma, Theta = group_power_network(p, K, M, seed=456)
    S, samples = sample_covariance_matrix(Sigma, N, seed=456)
    
    Sdict = dict()
    for k in np.arange(K):
        Sdict[k] = S[k,:,:].copy()
        
    G = construct_trivial_G(p, K)
    P = template_problem_MGL(Sdict, N, reg = 'GGL', latent = True, G = G)
    
    return

def test_GGL_ext_nonuniform():
    K = 4
    p = 20
    N = 200
    
    all_obs = dict()
    S = dict()
    for k in np.arange(K):
        X = np.random.rand(2+k,N)
        all_obs[k] = pd.DataFrame(X)
        S[k] = np.cov(all_obs[k], bias = True)
        
    ix_exist, ix_location = construct_indexer(list(all_obs.values()))
    G = create_group_array(ix_exist, ix_location, min_inst = 2)
    check_G(G, p)
    P = glasso_problem(S = S, N = N, reg = "GGL", reg_params = None, latent = False, G = G, do_scaling = True)
    reg_params = {'lambda1': 0.01, 'lambda2': 0.001}
    P.set_reg_params(reg_params)
    P.solve(verbose = True)
    
    return

###############################################################
### TEST SGL
###############################################################

def template_problem_SGL(S, N, latent = False):
    """
    template for testing the SGL problem object
    """
    P = glasso_problem(S = S, N = N, reg = None, latent = latent)
    print(P)
    

    reg_params = {'lambda1': 0.01}
    if latent:
        reg_params['mu1'] = 1.
    
    # set reg params and solve again
    P.set_reg_params(reg_params)
    P.solve()
    
    # test model selection    
    P.model_selection(modelselect_params = None, method = 'eBIC', gamma = 0.1)
    
    _ = P.solution.calc_ebic(gamma = 0.1)
    P.solution.calc_adjacency()
    return P

def test_SGL():
    Sigma, Theta = generate_precision_matrix(p, M=2, style = 'powerlaw', gamma = 2.8, prob = 0.1, seed = 1234)
    S, samples = sample_covariance_matrix(Sigma, N, seed = 1234)
    P = template_problem_SGL(S, N, latent = False)
    
    return
    
def test_SGL_latent():
    Sigma, Theta = generate_precision_matrix(p, M=2, style = 'powerlaw', gamma = 2.8, prob = 0.1, seed = 2345)
    S, samples = sample_covariance_matrix(Sigma, N, seed = 2345)
    P = template_problem_SGL(S, N, latent = True)
    
    return

################
### SCALING

def test_scaling_SGL():
    
    Sigma, Theta = generate_precision_matrix(p, M=2, style = 'powerlaw', gamma = 2.8, prob = 0.1, scale = True, seed = 789)
    S, samples = sample_covariance_matrix(Sigma, 10000, seed = 789)
    
    # create matrix with ones on diagonal
    np.fill_diagonal(S,1)
    
    sc = 1+np.random.rand(p)*10
    S2 = scale_array_by_diagonal(S, 1/sc)
    reg_params = {'lambda1': 0.1}
    
    solver_params = {'rho': 1., 'update_rho': False}
    
    # solve without scaling
    P = glasso_problem(S = S, N = N, reg = None, latent = False, do_scaling = False)
    P.set_reg_params(reg_params)
    P.solve(tol = 1e-15, rtol = 1e-15, solver_params = solver_params)
    
    # solve with scaling
    P2 = glasso_problem(S = S2, N = N, reg = None, latent = False, do_scaling = True)
    P2.set_reg_params(reg_params)
    P2.solve(tol = 1e-15, rtol = 1e-15, solver_params = solver_params)
    
    # precision is rescaled with 1/sc
    Theta = P.solution.precision_
    Theta2 = P2.solution.precision_
    
    Theta2 = scale_array_by_diagonal(Theta2, 1/sc) # from covariances to correlations on inverse
    
    assert_array_almost_equal(Theta, Theta2, decimal=3)
    assert_array_almost_equal(P.solution.adjacency_, P2.solution.adjacency_)
    
    return
    
##################
### LAMBDA1 MASK

def template_lambda1_mask_SGL(latent=False):
    p = 100
    N = 1000

    Sigma, Theta = generate_precision_matrix(p=p, M=2, style='erdos', gamma=2.8, prob=0.1, scale=False, seed=12345)
    S, samples = sample_covariance_matrix(Sigma, N, seed=12345)
    
    rng = np.random.RandomState(8787)
    lambda1_mask = 0.5 + 0.5*rng.rand(p,p)
    lambda1_mask = 0.5*(lambda1_mask + lambda1_mask.T)
    
    model_select_params= {'lambda1_mask': lambda1_mask, 'lambda1_range': np.logspace(-2,0,10)}

    P = glasso_problem(S = S, N = N, reg = None, latent = latent)
    P.set_modelselect_params(model_select_params)

    print(P.modelselect_params)
    P.model_selection(modelselect_params = None, method = 'eBIC', gamma = 0.1)    
    print(P.reg_params)
    
    if not latent:
        assert_almost_equal(P.reg_params['lambda1'], 0.1291549665014884)
    else:
         assert_almost_equal(P.reg_params['lambda1'], 0.1291549665014884)
         
    return

def test_lambda1_mask():
    template_lambda1_mask_SGL(latent=False)
    return

def test_lambda1_mask_latent():
    template_lambda1_mask_SGL(latent=True)
    return

