import numpy as np

from gglasso.helper.basic_linalg import trp

from gglasso.solver.admm_solver import ADMM_MGL

assert_tol = 1e-5

class glasso_problem:
    
    def __init__(self, S, reg, reg_params = None, latent = False):
        
        self.S = S
        
        self.latent = latent
        self.reg = reg
        
        # initialize and set regularization params
        self.reg_params = None
        self.set_reg_params(reg_params)
        
        self.derive_problem_formulation()
    
    def __repr__(self):
        
        return (
            " \n \nFORMULATION: "
            + "\n \n"
            + ("MULTIPLE" if self.multiple else "SINGLE")
            + " GRAPHICAL LASSO PROBLEM"
            + "\n")
        
    def derive_problem_formulation(self):
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
                self.check_covariance_3d()
                
            else:
                assert len(self.S.shape) == 2, f"The specified covariance data has shape {self.S.shape}, GGLasso can only handle 2 or 3dim-input"
                self.conforming = True
                self.check_covariance_2d()
                
                
        elif type(self.S) == list:
            
            assert len(self.S > 1), "Covariance data is a list with only one entry. This is a Single Graphical Lasso problem. Specify S as 2d-array."
            self.conforming = False
            self.multiple = True
            self.check_covariance_list()
            
        else:
            raise TypeError(f"Incorrect input type of S. You input {type(self.S)}, but np.ndarray or list is expected.")
    
    def check_covariance_3d(self):
        
        assert self.S.shape[1] == self.S.shape[2], f"Dimensions are not correct, 2nd and 3rd dimension have to match but shape is {self.S.shape}. Specify covariance data in format(K,p,p)!"
        
        assert np.max(np.abs(self.S - trp(self.S))) <= assert_tol, "Covariance data is not symmetric."
        
        (self.K,self.p,self.p) = self.S.shape
        
        return
    
    def check_covariance_2d(self):
        
        assert self.S.shape[0] == self.S.shape[1], f"Dimensions are not correct, 1st and 2nd dimension have to match but shape is {self.S.shape}. Specify covariance data in format(p,p)!"
        
        assert np.max(np.abs(self.S - self.S.T)) <= assert_tol, "Covariance data is not symmetric."
        
        (self.K,self.p) = self.S.shape
        
        return
    
    def check_covariance_list(self):
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
    
    def default_reg_params(self):
        reg_params_default = dict()
        reg_params_default['lambda1'] = 1e-3
        reg_params_default['lambda2'] = 1e-3
        
        if self.latent:
            reg_params_default['mu1'] = 1e-3*np.ones(self.K)
        else:
            reg_params_default['mu1'] = None
            
        return reg_params_default
        
    def set_reg_params(self, reg_params = None):
        """
        Parameters
        ----------
        reg_params : dict
            
        Returns
        -------
        None.

        """
        if reg_params is None:
            reg_params = dict()
        else:
            assert type(reg_params) == dict
        
        # when initialized set to default
        if self.reg_params is None:
            self.reg_params = self.default_reg_params()
        
        
        # update with given input
        # update with empty dict does not change the dictionary
        self.reg_params.update(reg_params)
            
        return
    
    def default_start_point(self):
        
        if not self.multiple:
            X = np.eye(self.p)
            
        elif self.conforming:
            X = np.repeat(np.eye(self.p)[np.newaxis,:,:], self.K, axis=0)
        
        else:
            X = dict()
            for k in range(self.K):
                X[k] = np.eye(self.p[k])
                
        return X
            
    
    def set_start_point(self, Omega_0 = None):
        
        if Omega_0 is not None:
            self.Omega_0 = Omega_0.copy()
        else:
            self.Omega_0 = self.default_start_point()
            
        return
        
    def default_solver_params(self):
        
        solver_params = dict()
        solver_params['verbose'] = False
        solver_params['measure'] = False
        solver_params['rho'] = 1.
        solver_params['max_iter'] = 1000
        solver_params['eps_admm'] = 1e-5
        
        return solver_params
        
      
    def solve(self, Omega_0 = None, solver_params = None):
        
        self.set_start_point(Omega_0)
        
        
        if self.conforming:
            
            sol, info = ADMM_MGL(S = self.S, lambda1 = self.reg_params['lambda_1'], lambda2 = self.reg_params['lambda_2'], reg = self.reg,\
                     Omega_0 = self.Omega_0, latent = self.latent(), mu1 = self.reg_params['mu_1'], **solver_params)
            
                
                
        else:
            1
    
        return

    