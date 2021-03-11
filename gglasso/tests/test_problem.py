import numpy as np
import time

from gglasso.helper.data_generation import time_varying_power_network, group_power_network,sample_covariance_matrix
from gglasso.problem import glasso_problem
from gglasso.helper.ext_admm_helper import construct_trivial_G


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
    
    modelselectparams = dict()
    modelselectparams['lambda1_range'] = np.logspace(-3,0,4)
    modelselectparams['w2_range'] = np.logspace(-1,-3,3)
              
    if latent:
        modelselectparams['mu1_range'] = np.logspace(-2,0,4)
    else:
        modelselectparams['mu1_range'] = None
    
    P.model_selection(modelselect_params = modelselectparams, method = 'eBIC', gamma = 0.1)
    #tmp = P.modelselect_stats.copy()
    P.solution.calc_ebic(gamma = 0.1)
    return

def test_GGL():
    Sigma, Theta = group_power_network(p, K, M)    
    S, samples = sample_covariance_matrix(Sigma, N)
    template_problem_MGL(S, N, reg = 'GGL', latent = False)   
    return

def test_GGL_latent():
    Sigma, Theta = group_power_network(p, K, M)    
    S, samples = sample_covariance_matrix(Sigma, N)
    template_problem_MGL(S, N, reg = 'GGL', latent = True)
    return

def test_FGL():
    Sigma, Theta = time_varying_power_network(p, K, M)
    S, samples = sample_covariance_matrix(Sigma, N)
    template_problem_MGL(S, N, reg = 'FGL', latent = False)
    return

def test_FGL_latent():
    Sigma, Theta = time_varying_power_network(p, K, M)
    S, samples = sample_covariance_matrix(Sigma, N)
    template_problem_MGL(S, N, reg = 'FGL', latent = True)
    return

def test_GGL_ext():
    Sigma, Theta = group_power_network(p, K, M)
    S, samples = sample_covariance_matrix(Sigma, N)
    
    Sdict = dict()
    for k in np.arange(K):
        Sdict[k] = S[k,:,:].copy()
        
    G = construct_trivial_G(p, K)
    template_problem_MGL(S, N, reg = 'GGL', latent = False, G = G)
    return

def test_GGL_ext_latent():
    Sigma, Theta = group_power_network(p, K, M)
    S, samples = sample_covariance_matrix(Sigma, N)
    
    Sdict = dict()
    for k in np.arange(K):
        Sdict[k] = S[k,:,:].copy()
        
    G = construct_trivial_G(p, K)
    template_problem_MGL(S, N, reg = 'GGL', latent = True, G = G)
    return

###############################################################
### TEST SGL
###############################################################

def template_problem_SGL(S, N, latent = False):
    """
    template for testing the SGL problem object
    """
    P = glasso_problem(S = S, N = N, reg = None, latent = latent)
    
    modelselectparams = dict()
    modelselectparams['lambda1_range'] = np.logspace(-3,0,4)
    
    if latent:
        modelselectparams['mu1_range'] = np.logspace(-2,0,4)
    else:
        modelselectparams['mu1_range'] = None
    
    P.model_selection(modelselect_params = modelselectparams, method = 'eBIC', gamma = 0.1)
    #tmp = P.modelselect_stats.copy()
    P.solution.calc_ebic(gamma = 0.1)
    return

def test_SGL():
    Sigma, Theta = group_power_network(p, K = 1, M = 2)
    S, samples = sample_covariance_matrix(Sigma, N); S = S[0,:,:]  
    template_problem_SGL(S, N, latent = False)
    return
    
def test_SGL_latent():
    Sigma, Theta = group_power_network(p, K = 1, M = 2)
    S, samples = sample_covariance_matrix(Sigma, N); S = S[0,:,:]  
    template_problem_SGL(S, N, latent = True)
    return
        