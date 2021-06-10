"""
author: Fabian Schaipp
"""
import numpy as np

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
    print(P)
    
    modelselectparams = dict()
    modelselectparams['lambda1_range'] = np.logspace(-3,0,4)
    modelselectparams['lambda2_range'] = np.logspace(-1,-3,3)
              
    if latent:
        modelselectparams['mu1_range'] = np.logspace(-2,0,4)
    else:
        modelselectparams['mu1_range'] = None
    
    
    P.solve()
    reg_params = {'lambda1': 0.01, 'lambda2': 0.001}
    if latent:
        reg_params['mu1'] = 1.
    
    # set reg params and solve again
    P.set_reg_params(reg_params)
    P.solve()
    
    # test model selection
    P.model_selection(method = 'AIC')
    P.model_selection(modelselect_params = modelselectparams, method = 'eBIC', gamma = 0.1)
    
    #tmp = P.modelselect_stats.copy()
    _ = P.solution.calc_ebic(gamma = 0.1)
    P.solution.calc_adjacency()
    
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
    print(P)
    
    
    P.solve()
    reg_params = {'lambda1': 0.01}
    if latent:
        reg_params['mu1'] = 1.
    
    # set reg params and solve again
    P.set_reg_params(reg_params)
    P.solve()
    
    # test model selection
    # modelselectparams = dict()
    # modelselectparams['lambda1_range'] = np.logspace(-3,0,4)
    
    # if latent:
    #     modelselectparams['mu1_range'] = np.logspace(-2,0,4)
    # else:
    #     modelselectparams['mu1_range'] = None
    
    P.model_selection(method = 'AIC')
    P.model_selection(modelselect_params = None, method = 'eBIC', gamma = 0.1)
    
    #tmp = P.modelselect_stats.copy()
    _ = P.solution.calc_ebic(gamma = 0.1)
    P.solution.calc_adjacency()
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
        

def test_scaling_SGL():
    
    Sigma, Theta = group_power_network(p, K = 1, M = 2)
    S, samples = sample_covariance_matrix(Sigma, N); S = S[0,:,:]  
    
    S2 = 10*S
    reg_params = {'lambda1': 0.01}
    reg_params2 = {'lambda1': 0.1}
    
    P = glasso_problem(S = S2, N = N, reg = None, latent = False, do_scaling = True)
    P.set_reg_params(reg_params)
    P.solve(tol = 1e-10, rtol = 1e-15)
    
    solver_params = {'rho': 10}
        
    P2 = glasso_problem(S = S2, N = N, reg = None, latent = False, do_scaling = False)
    P2.set_reg_params(reg_params2)
    P2.solve(tol = 1e-10, rtol = 1e-15, solver_params = solver_params)    
    
    
    Theta = P.solution.precision_
    Theta2 = P2.solution.precision_
    
    assert np.linalg.norm(Theta - Theta2)/np.linalg.norm(Theta) <= 0.02
    
    return
    