import numpy as np

from .helper.basic_linalg import trp, adjacency_matrix, scale_array_by_diagonal
from .helper.ext_admm_helper import check_G

from .solver.admm_solver import ADMM_MGL
from .solver.single_admm_solver import ADMM_SGL
from .solver.ext_admm_solver import ext_ADMM_MGL

from .helper.model_selection import grid_search, single_grid_search, K_single_grid, ebic, ebic_single


assert_tol = 1e-5

class glasso_problem:
    
    def __init__(self, S, N, reg = "GGL", reg_params = None, latent = False, G = None, do_scaling = True):
        
        self.S = S.copy()
        self.N = N
        self.latent = latent
        self.G = G
        self.do_scaling = do_scaling
        
        self._derive_problem_formulation()
        
        # initialize and set regularization params
        self.reg_params = None
        self.modelselect_params = None
        self.set_reg_params(reg_params)
        
        if self.multiple:
            assert reg in ["GGL", "FGL"], "Specify 'GGL' for Group Graphical Lasso or 'FGL' for Fused Graphical Lasso (or None for Single Graphical Lasso)"
            self.reg = reg
        else:
            self.reg = None
            
            
        # create an instance of GGLassoEstimator (before scaling S!)
        self.solution = GGLassoEstimator(S = self.S.copy(), N = self.N, p = self.p, K = self.K,\
                         multiple = self.multiple, latent = self.latent, conforming = self.conforming)
        
        
        # scale S by diagonal
        if self.do_scaling:
            self._scale_input_to_correlation()
        
        return
        
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
        
        (self.p,self.p) = self.S.shape
        self.K = 1
        
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
    #### SCALING
    ##############################################
    
    def _rescale_to_covariances(self, X, scale):
        """
        rescales X with the given scale
        X: object of type like input data S
        scale: array with diagonal elements of unscaled input S --> use self._scaled
        """
        Y = X.copy()
        if not self.multiple:
            Y = scale_array_by_diagonal(X, d = scale)
        else:
            for k in range(self.K):
                Y[k] = scale_array_by_diagonal(X[k], d = scale[k])
                
        return Y
    
    def _scale_input_to_correlation(self):
        """
        scales input data S by diagonal elements 
        scale factors are stored in self._scale for rescaling later
        
        NOTE: this overwrites self.S!
        """
        
        print("NOTE: input data S is rescaled with the diagonal elements, this has impact on the scale of the regularization parameters!")
        
        if not self.multiple:
            self._scale = np.diag(self.S)
            self.S = scale_array_by_diagonal(self.S)
        else:
            self._scale = np.vstack([np.diag(self.S[k]) for k in range(self.K)])
            for k in range(self.K):
                self.S[k] = scale_array_by_diagonal(self.S[k])
        return
    

    
    ##############################################
    #### DEFAULT PARAMETERS
    ##############################################
    def _default_reg_params(self):
        reg_params_default = dict()
        
        reg_params_default['lambda1'] = 1e-2
        if self.multiple:
            reg_params_default['lambda2'] = 1e-2
        if self.latent:
            if self.multiple:
                reg_params_default['mu1'] = 1e-1*np.ones(self.K)
            else:
                reg_params_default['mu1'] = 1e-1
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
        """

        Parameters
        ----------
        Omega_0 : array or dict, optional
            Starting point for solver. Needs to be of same type as input data S. The default is None.

        Returns
        -------
        None.

        """
        
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
         
        self.solver_params = self._default_solver_params()
        self.solver_params.update(solver_params)
        
        #print(self.solver_params.keys())
        
        print(f"\n Solve problem with {solver.upper()} solver... \n ")
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
                
 
        # rescale
        if self.do_scaling:
            #print("Diagonal of solution before rescaling:", np.diag(sol['Theta']))
            sol['Theta'] = self._rescale_to_covariances(sol['Theta'], self._scale)
        
            
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
        params['lambda1_range'] = np.logspace(-3,0,10)
        if self.multiple:
            params['w2_range'] = np.logspace(-1,-3,5)
            #params['lambda2_range'] = np.logspace(-3,1,10)
            
        if self.latent:
            params['mu1_range'] = np.logspace(-2,0,10)
        else:
            params['mu1_range'] = None
        
        return params
    
    def set_modelselect_params(self, modelselect_params = None):
        """
        modelselect_params : dict
            Contains values for (a subset of) the grid parameters for lambda1, lambda2, mu1
        """
        if modelselect_params is None:
            modelselect_params = dict()
        else:
            assert type(modelselect_params) == dict
        
        # when initialized set to default
        if self.modelselect_params is None:
            print("NOTE: No grid for model selection is specified and thus default values are used. A grid can be specified with the argument modelselect_params.")
            self.modelselect_params = self._default_modelselect_params()
        
        
        # update with given input
        # update with empty dict does not change the dictionary
        self.modelselect_params.update(modelselect_params)
            
        return

    def model_selection(self, modelselect_params = None, method = 'eBIC', gamma = 0.1):
        
        assert (gamma >= 0) and (gamma <= 1), "gamma needs to be chosen as a parameter in [0,1]."
        assert method in ['eBIC', 'AIC'], "Supported evaluation methods are eBIC and AIC."
        
        self.set_modelselect_params(modelselect_params)
        
        ###############################
        # SINGLE GL --> GRID SEARCH lambda1/mu
        ###############################
        if not self.multiple:
            sol, _, _, stats = single_grid_search(S = self.S, lambda_range = self.modelselect_params['lambda1_range'], N = self.N, \
                               method = method, gamma = gamma, latent = self.latent, mu_range = self.modelselect_params['mu1_range'])
            
            # update the regularization parameters to the best grid point
            self.set_reg_params(stats['BEST'])
           
        else:
            # choose solver 
            if self.conforming:
                solver = ADMM_MGL
            else:
                solver = ext_ADMM_MGL
            
            ###############################
            # LATENT VARIABLES --> FIRST STAGE lambda1/mu1 for each instance
            ############################### 
            if self.latent:
                est_uniform, est_indv, stage1_statistics = K_single_grid(S = self.S, lambda_range = self.modelselect_params['lambda1_range'], N = self.N, method = method,\
                                                                  gamma = gamma, latent = self.latent, mu_range = self.modelselect_params['mu1_range'])            
                
                ix_mu = stage1_statistics['ix_mu']
                
                # store results from stage 1 (may be needed to compare Single estimator vs. Joint estimator)
                self.est1 = est_uniform
                self.est2 = est_indv
                self.stage1_stats = stage1_statistics          
            else:
                ix_mu = None
            
            
            ###############################
            # SECOND STAGE --> GRID SEARCH lambda1/lambda2
            ############################### 
    
            stats, _, sol = grid_search(solver, S = self.S, N = self.N, p = self.p, reg = self.reg, l1 = self.modelselect_params['lambda1_range'], \
                                        l2 = None, w2 = self.modelselect_params['w2_range'], method= method, gamma = gamma, \
                                        G = self.G, latent = self.latent, mu_range = self.modelselect_params['mu1_range'], ix_mu = ix_mu, verbose = False)
            
            # update the regularization parameters to the best grid point
            self.set_reg_params(stats['BEST'])
        
            
        ###############################
        # SET SOLUTION AND STORE INFOS
        ###############################
            
        # rescale
        if self.do_scaling:
            #print("Diagonal of solution before rescaling:", np.diag(sol['Theta']))
            sol['Theta'] = self._rescale_to_covariances(sol['Theta'], self._scale)
        
            
        # set the computed solution
        if self.latent:
            self.solution._set_solution(Theta = sol['Theta'], L = sol['L'])   
        else:
            self.solution._set_solution(Theta = sol['Theta'])   
        
        
        self.modelselect_stats = stats.copy()
        
        
        return
    
    
    

    
#%%
from sklearn.base import BaseEstimator

class GGLassoEstimator(BaseEstimator):
    
    def __init__(self, S, N, p, K, multiple = True, latent = False, conforming = True):
        
        self.multiple = multiple
        self.latent = latent
        self.conforming = conforming
        self.K = K
        
        self.n_samples = N
        self.n_features = p
        
        self.precision_ = None
        self.sample_covariance_ = S.copy()
        self.lowrank_ = None
        
        
        super(GGLassoEstimator, self).__init__()
        
        self.adjacency_ = None
        self.ebic_ = None
        
        
        return
    
    def _set_solution(self, Theta, L = None):
        
        self.precision_ = Theta.copy()
        self.calc_adjacency()
        
        if L is not None:
            self.lowrank_ = L.copy()
        
        return
    
    def calc_ebic(self, gamma = 0.5):
        
        if self.multiple:
            self.ebic_ = ebic(self.S, self.precision_, self.n_samples, gamma = gamma)
            
        else:
            self.ebic_ = ebic_single(self.sample_covariance_, self.precision_, self.n_samples, gamma = gamma)        
        
        return self.ebic_
    
    def calc_adjacency(self):
        
        if self.conforming:
            self.adjacency_ = adjacency_matrix(S = self.precision_, t = 1e-5)
        
        else:
            self.adjacency_ = dict()
            for k in range(self.K):
                self.adjacency_[k] = adjacency_matrix(S = self.precision_[k], t = 1e-5)
            
        return
        






