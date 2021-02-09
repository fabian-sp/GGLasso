import numpy as np

from .helper.basic_linalg import trp
from .helper.ext_admm_helper import check_G

from .solver.admm_solver import ADMM_MGL
from .solver.single_admm_solver import ADMM_SGL
from .solver.ext_admm_solver import ext_ADMM_MGL

from .helper.model_selection import ebic, ebic_single


assert_tol = 1e-5

class glasso_problem:
    
    def __init__(self, S, N, reg = "GGL", reg_params = None, latent = False, G = None):
        
        self.S = S
        self.N = N
        self.latent = latent
        
        self.G = G
        
        # initialize and set regularization params
        self.reg_params = None
        self.set_reg_params(reg_params)
        
        self._derive_problem_formulation()
        
        if self.multiple:
            assert reg in ["GGL", "FGL"], "Specify 'GGL' for Group Graphical Lasso or 'FGL' for Fused Graphical Lasso (or None for Single Graphical Lasso)"
            self.reg = reg
        else:
            self.reg = None
            
        
    def __repr__(self):
        
        if self.multiple:
            prefix = ("FUSED" if self.reg == "FGL" else "GROUP")
        else:
            prefix = "SINGLE"
        return (
            " \n"
            + prefix
            + " GRAPHICAL LASSO PROBLEM"
            + "\n"
            + "Regularization parameters:\n"
            + f"{self.reg_params}"
            )
        
    def _derive_problem_formulation(self):
        """
        - derives the problem formulation type from the given input of covariance matrices
        - sets the problems dimensions (K,p_k)
        - checks the input data (e.g. symmetry)
        """
        
        self.conforming = False
        self.multiple = False
        
        if type(self.S) == np.ndarray:
            
            if len(self.S.shape) == 3:
                self.conforming = True
                self.multiple = True
                self._check_covariance_3d()
                
            else:
                assert len(self.S.shape) == 2, f"The specified covariance data has shape {self.S.shape}, GGLasso can only handle 2 or 3dim-input"
                self.conforming = True
                self._check_covariance_2d()
                
                
        elif type(self.S) == list:
            
            assert len(self.S > 1), "Covariance data is a list with only one entry. This is a Single Graphical Lasso problem. Specify S as 2d-array."
            assert self.G is not None, "For non-conforming dimensions, the input G has to be specified for bookeeping the overlapping variables."
            
            self.conforming = False
            self.multiple = True
            self._check_covariance_list()
            
            # G is also checked in the solver
            check_G(self.G, self.p)
            
        else:
            raise TypeError(f"Incorrect input type of S. You input {type(self.S)}, but np.ndarray or list is expected.")
    
    ##############################################
    #### CHECK INPUT DATA
    ##############################################
    def _check_covariance_3d(self):
        
        assert self.S.shape[1] == self.S.shape[2], f"Dimensions are not correct, 2nd and 3rd dimension have to match but shape is {self.S.shape}. Specify covariance data in format(K,p,p)!"
        
        assert np.max(np.abs(self.S - trp(self.S))) <= assert_tol, "Covariance data is not symmetric."
        
        (self.K,self.p,self.p) = self.S.shape
        
        return
    
    def _check_covariance_2d(self):
        
        assert self.S.shape[0] == self.S.shape[1], f"Dimensions are not correct, 1st and 2nd dimension have to match but shape is {self.S.shape}. Specify covariance data in format(p,p)!"
        
        assert np.max(np.abs(self.S - self.S.T)) <= assert_tol, "Covariance data is not symmetric."
        
        (self.K,self.p) = self.S.shape
        
        return
    
    def _check_covariance_list(self):
        self.K = len(self.S)
        
        self.p = np.zeros(self.K, dtype = int)
        
        S_dict = dict()
        
        for k in range(self.K):
            assert self.S[k].shape[0] == self.S[k].shape[1], f"Dimensions are not correct, 1st and 2nd dimension have to match but do not match for instance {k}."
            assert np.max(np.abs(self.S[k] - self.S[k].T)) <= assert_tol, f"Covariance data for instance {k} is not symmetric."
            
            self.p[k]= self.S[k].shape[0]
            
            S_dict[k] = self.S[k].copy()
        
            
        # S gets converted from alist to a dict with keys 1,..,K
        self.S = S_dict
        
        return
    
    ##############################################
    #### DEFAULT PARAMETERS
    ##############################################
    def _default_reg_params(self):
        reg_params_default = dict()
        reg_params_default['lambda1'] = 1e-3
        reg_params_default['lambda2'] = 1e-3
        
        if self.latent:
            if self.multiple:
                reg_params_default['mu1'] = 1e-3*np.ones(self.K)
            else:
                reg_params_default['mu1'] = 1e-3
        else:
            reg_params_default['mu1'] = None
            
        return reg_params_default
    
    def _default_start_point(self):
        
        if not self.multiple:
            X = np.eye(self.p)
            
        elif self.conforming:
            X = np.repeat(np.eye(self.p)[np.newaxis,:,:], self.K, axis=0)
        
        else:
            X = dict()
            for k in range(self.K):
                X[k] = np.eye(self.p[k])
                
        return X
    
    def _default_solver_params(self):
        
        solver_params = dict()
        solver_params['verbose'] = False
        solver_params['measure'] = False
        solver_params['rho'] = 1.
        solver_params['max_iter'] = 1000
        
        return solver_params
        
    def set_reg_params(self, reg_params = None):
        """
        reg_params : dict
            Contains values for (a subset of) the regularization parameters lambda1, lambda2, mu1
        """
        if reg_params is None:
            reg_params = dict()
        else:
            assert type(reg_params) == dict
        
        # when initialized set to default
        if self.reg_params is None:
            self.reg_params = self._default_reg_params()
        
        
        # update with given input
        # update with empty dict does not change the dictionary
        self.reg_params.update(reg_params)
            
        return

    
    def set_start_point(self, Omega_0 = None):
        
        if Omega_0 is not None:
            # TODO: check if Omega_0 has correct type (depends on problem parameters)
            self.Omega_0 = Omega_0.copy()
        else:
            self.Omega_0 = self._default_start_point()
            
        return
        
    ##############################################
    #### SOLVING
    ##############################################
      
    def solve(self, Omega_0 = None, solver_params = dict(), tol = 1e-4, solver = 'admm'):
        
        assert solver in ["admm", "ppdna"], "There are two solver types supported, ADMM and PPDNA. Specify the argument solver = 'admm' or solver = 'ppdna'."
        
        if solver == "ppdna":
            assert self.multiple,"PPDNA solver is only supported for MULTIPLE Graphical Lassp problems."
            assert not self.latent, "PPDNA solver is only supported for problems without latent variables."
            assert self.conforming, "PPDNA solver is only supported for problems with conforming dimensions."
        
        
        self.set_start_point(Omega_0)
        self.tol = tol
        
        #forbidden_keys = ['lambda1', 'lambda2', 'mu1', 'latent', 'Omega_0', 'reg']
        
        self.solver_params = self._default_solver_params().update(solver_params)
        
        print(solver_params.keys())
        
        print(f"Solve problem with {solver} solver...")
        if not self.multiple:
            sol, info = ADMM_SGL(S = self.S, lambda1 = self.reg_params['lambda1'], Omega_0= self.Omega_0, \
                                 eps_admm = self.tol , latent = self.latent, mu1 = self.reg_params['mu1'], **self.solver_params)
                
                
        elif self.conforming:
            
            sol, info = ADMM_MGL(S = self.S, lambda1 = self.reg_params['lambda1'], lambda2 = self.reg_params['lambda2'], reg = self.reg,\
                     Omega_0 = self.Omega_0, latent = self.latent, mu1 = self.reg_params['mu1'],\
                         eps_admm = self.tol, **self.solver_params)
            
                
                
        else:
            sol, info = ext_ADMM_MGL(S = self.S, lambda1 = self.reg_params['lambda1'], lambda2 = self.reg_params['lambda2'], reg = self.reg,\
                                     Omega_0 = self.Omega_0, G = self.G, eps_admm = self.tol,\
                                         latent = self.latent, mu1 = self.reg_params['mu1'], **self.solver_params)
                
        
        # create an instance of GGLassoEstimator
        self.solution = GGLassoEstimator(S = self.S, N = self.N, p = self.p, \
                         multiple = self.multiple, latent = self.latent, conforming = self.conforming)
        
        # set the computed solution
        if self.latent:
            self.solution._set_solution(Theta = sol['Theta'], L = sol['L'])   
        else:
            self.solution._set_solution(Theta = sol['Theta'])   
        
        self.solver_info = info.copy()
        return


    ##############################################
    #### MODEL SELECTION
    ##############################################
    
    def _default_modelselect_params(self):
        
        params = dict()
        params['lambda1_range'] = np.logspace(-3,1,10)
        if self.multiple:
            params['w2_range'] = np.logspace(-1,-4,5)
            #params['lambda2_range'] = np.logspace(-3,1,10)
            
        if self.latent:
            params['mu1_range'] = np.logspace(-1,1,10)
        
        
        return params
    
    def set_modelselect_params(self, modelselect_params = None):
        """
        params : dict
            Contains values for (a subset of) the grid parameters for lambda1, lambda2, mu1
        """
        if modelselect_params is None:
            modelselect_params = dict()
        else:
            assert type(modelselect_params) == dict
        
        # when initialized set to default
        if self.modelselect_params is None:
            self.modelselect_params = self._default_modelselect_params()
        
        
        # update with given input
        # update with empty dict does not change the dictionary
        self.modelselect_params.update(modelselect_params)
            
        return
    
#%%

from sklearn.base import BaseEstimator


class GGLassoEstimator(BaseEstimator):
    
    def __init__(self, S, N, p, multiple = True, latent = False, conforming = True):
        
        self.multiple = multiple
        self.latent = latent
        self.conforming = conforming
        
        self.n_samples = N
        self.n_features = p
        
        self.precision_ = None
        self.sample_covariance_ = S.copy()
        self.lowrank_ = None
        
        
        super(GGLassoEstimator, self).__init__()
        
        return
    
    def _set_solution(self, Theta, L = None):
        
        self.precision_ = Theta.copy()
        self.lowrank_ = L.copy()
        
        return
    
    def ebic(self, gamma = 0.5):
        
        if self.mutliple:
            self.ebic_ = ebic(self.S, self.precision_, self.n_samples, gamma = gamma)
            
        else:
            self.ebic_ = ebic_single(self.S, self.precision_, self.n_samples, gamma = gamma)        
        
        






