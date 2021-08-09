"""
author: Fabian Schaipp
"""
import numpy as np

from .helper.basic_linalg import trp, adjacency_matrix, scale_array_by_diagonal
from .helper.ext_admm_helper import check_G

from .solver.admm_solver import ADMM_MGL
from .solver.single_admm_solver import ADMM_SGL
from .solver.ext_admm_solver import ext_ADMM_MGL

from .helper.model_selection import grid_search, single_grid_search, K_single_grid, ebic, ebic_single


assert_tol = 1e-5

class glasso_problem:
    """
    Class for Graphical Lasso problems. After solving, you can access the estimators with ``self.solution``. See documentation of ``GGLassoEstimator`` for details.
    
    Important attributes which determine the problem type:
        * ``self.multiple``: specifies if SGL or MGL.
        * ``self.latent``: specifies if latent variables are modeled.
        * ``self.reg``: specifies if FGL or GGL (if MGL).
        * ``self.conforming``: specifies if all variables are present in all instances (for GGL).
        
    An instance of this class can be printed in order to inspect the derived problem formulation.    
    
    Parameters
    ----------
    S : 2d/3d-array or list/dict
        Empirical covariance matrices.
        
        * For SGL, use 2d array of shape (p,p). 
        * For MGL use 3d array of shape (K,p,p). 
        * For GGL with non-conforming dimensions, use a list/dict of length K. Will be transformed to a dict with keys 1,..K.
        
        For MGL, each ``S[k]`` has to be symmetric and positive semidefinite.
        Note: scaling ``S`` to correlations might be helpful, see option ``do_scaling``.
        
    N : int or integer array of length K
        Number of samples for each instance k=1,..,K.
        
    reg : str, optional
        Type of regularization for MGL problems.
        
        * 'FGL' = Fused Graphical Lasso
        * 'GGL' = Group Graphical Lasso
        
        The default is 'GGL'.
        
    reg_params : dict, optional
        Dictionary of regularization parameters. Possible keys are:
            
        * ``'lambda1'``: float, positive
        * ``'lambda2'``: float, positive
        * ``'mu1'``: float or array of length K, positive. Only needed if ``latent = True``.
               
    latent : boolean, optional
        Specify whether latent variables should be modeled.
        
        * ``latent = True``: inverse covariance is assumed to have form :math:`\Theta-L` (sparse - low rank).
        * ``latent = False``: inverse covariance is assumed to have form :math:`\Theta` (sparse).
        
        The default is False.
        
    G : 3d-array of shape(2,L,K), optional
        Only needed when dimensions are non-conforming, i.e. if number of variables is different in each instance.
        See :ref:`Nonconforming GGL` on how to create G.
        
    do_scaling : boolean, optional
        Whether to scale input S to correlations. The default is ``False``.
        If ``True``, the output is re-scaled to covariances after solving. 

    """
    
    def __init__(self, S, N, reg = "GGL", reg_params = None, latent = False, G = None, do_scaling = False):
        
        self.S = S.copy()
        self.N = N
        self.latent = latent
        self.G = G
        self.do_scaling = do_scaling
        
        self._derive_problem_formulation()
        
        # initialize and set regularization params
        self.reg_params = None
        self.modelselect_params = self._default_modelselect_params()
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
        if self.latent:
            suffix = "WITH LATENT VARIABLES"
        else:
            suffix = ""
        return (
            " \n"
            + prefix + " GRAPHICAL LASSO PROBLEM " + suffix
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
                
                
        elif type(self.S) == list or type(self.S) == dict:
            
            assert len(self.S) > 1, "Covariance data is a list/dict with only one entry. This is a Single Graphical Lasso problem. Specify S as 2d-array."
            assert self.G is not None, "For non-conforming dimensions, the input G has to be specified for bookeeping the overlapping variables."
            
            self.conforming = False
            self.multiple = True
            self._check_covariance_list()
            
            # G is also checked in the solver
            check_G(self.G, self.p)
            
        else:
            raise TypeError(f"Incorrect input type of S. You input {type(self.S)}, but np.ndarray or list/dict is expected.")
    
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
        scale: array (or list) with diagonal elements of unscaled input S --> use self._scale
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
            self._scale = [np.diag(self.S[k]) for k in range(self.K)]
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
        
        if self.conforming or not self.multiple:
            solver_params['update_rho'] = True
        
        return solver_params
        
    def set_reg_params(self, reg_params = None):
        """
        Sets/updates the regularization parameters for the problem.
        
        Parameters
        ----------
        reg_params : dict, optional
        
            Possible keys:
                * ``'lambda1'``: float, positive
                * ``'lambda2'``: float, positive
                * ``'mu1'``: float or array of length K, positive. Only needed if ``latent = True``.
            
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
        Set the starting point for solving the problem.
        
        Parameters
        ----------
        Omega_0 : array or dict, optional
            Starting point for solver. Needs to be of same type as input data S. The default is None.

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
      
    def solve(self, Omega_0 = None, solver_params = dict(), tol = 1e-8, rtol = 1e-7, solver = 'admm'):
        """
        Method for solving the Graphical Lasso problem formulation.
        After solving, an instance of ``GGLassoEstimator`` will be created and assigned to ``self.solution``.
        
        Parameters
        ----------
        Omega_0 : 2d/3d-array or dict, optional
            Start point for solver. If not specified, identity matrix is used as a starting point.
            
            * For SGL, specifiy a symmetric 2d-array of shape (p,p).
            * For MGL, specifiy a symmetric (for each k) 3d-array of shape (K,p,p).
            * For non-conforming MGL, specifiy a dictionary with keys 1,...,K and symmetric 2d-arrays of shape :math:`(p_k,p_k)` as values.
        
        solver_params : dict, optional
            Parameters for the solvers. Is given as kwargs for the solver. See doc of the solvers for more details.
            
        tol : float, optional
            Tolerance for solving. The smaller it is, the longer it will take to solve the problem. 
            The default is 1e-5.
            
        rtol : float, optional
            Relative Tolerance for solving. The smaller it is, the longer it will take to solve the problem. 
            The default is 1e-4.
            
        solver : str, optional
            Solver name. At this point, we use ADMM for all formulations.
            The default is 'admm'.

        Returns
        -------
        None.

        """
        
        assert solver in ["admm"], "Currently only the ADMM solver is supported as it is implemented for all cases."
        
        # if solver == "ppdna":
        #     assert self.multiple,"PPDNA solver is only supported for MULTIPLE Graphical Lassp problems."
        #     assert not self.latent, "PPDNA solver is only supported for problems without latent variables."
        #     assert self.conforming, "PPDNA solver is only supported for problems with conforming dimensions."
        
        
        self.set_start_point(Omega_0)
        self.tol = tol
        self.rtol = rtol
         
        self.solver_params = self._default_solver_params()
        self.solver_params.update(solver_params)
        
        #print(self.solver_params.keys())
        
        print(f"\n Solve problem with {solver.upper()} solver... \n ")
        if not self.multiple:
            sol, info = ADMM_SGL(S = self.S, lambda1 = self.reg_params['lambda1'], Omega_0= self.Omega_0, \
                                 tol = self.tol , rtol = self.rtol, latent = self.latent, mu1 = self.reg_params['mu1'], **self.solver_params)
                
                
        elif self.conforming:         
            sol, info = ADMM_MGL(S = self.S, lambda1 = self.reg_params['lambda1'], lambda2 = self.reg_params['lambda2'], reg = self.reg,\
                     Omega_0 = self.Omega_0, latent = self.latent, mu1 = self.reg_params['mu1'],\
                         tol = self.tol, rtol = self.rtol, **self.solver_params)
            
                               
        else:
            sol, info = ext_ADMM_MGL(S = self.S, lambda1 = self.reg_params['lambda1'], lambda2 = self.reg_params['lambda2'], reg = self.reg,\
                                     Omega_0 = self.Omega_0, G = self.G, tol = self.tol, rtol = self.rtol,\
                                         latent = self.latent, mu1 = self.reg_params['mu1'], **self.solver_params)
                
 
        # rescale
        if self.do_scaling:
            sol['Theta'] = self._rescale_to_covariances(sol['Theta'], self._scale)
            if self.latent:
                sol['L'] = self._rescale_to_covariances(sol['L'], self._scale)
        
            
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
        params['lambda1_range'] = np.logspace(0,-3,5)
        if self.multiple:
            #params['w2_range'] = np.logspace(-1,-3,5)
            params['lambda2_range'] = np.logspace(-1,-4,4)
            
        if self.latent:
            params['mu1_range'] = np.logspace(0,-2,5)
        else:
            params['mu1_range'] = None
        
        return params
    
    def set_modelselect_params(self, modelselect_params = None):
        """
        Set the ranges of regularization parameters for the grid searches.
        
        Parameters
        ----------
        modelselect_params : dict
            Contains values for (a subset of) the grid parameters for :math:`\lambda_1`, :math:`\lambda_2`, :math:`\mu_1`.
            Each dictionary value should be an array. For optimal performance sort :math:`\lambda_1` in a descending order.
            
            Possible dictionary keys:
                * ``'lambda1_range'``: range for :math:`\lambda_1` parameter.
                * ``'lambda2_range'``: range for :math:`\lambda_2` parameter.
                * ``'mu1_range'``: range for :math:`\mu_1` parameter.
                
        """
        
        if modelselect_params is None:
            modelselect_params = dict()
            print("NOTE: No grid for model selection is specified and thus default values are used. A grid can be specified with the argument modelselect_params.")
            
        else:
            assert type(modelselect_params) == dict
        
        # update with given input
        # update with empty dict does not change the dictionary
        self.modelselect_params.update(modelselect_params)
            
        return

    def model_selection(self, modelselect_params = None, method = 'eBIC', gamma = 0.1, tol = 1e-7, rtol = 1e-7):
        """
        Method for doing model selection, i.e. trying to find the best regularization parameters.
        An instance of ``GGLassoEstimator`` will be created and assigned to ``self.solution``.
        
        Strategy for the different problem formulations:
            
        * SGL: solve on a path of :math:`\lambda_1` values or on a grid of :math:`(\lambda_1, \mu_1)` values if ``latent=True``. Choose the grid point where the eBIC is minimal.
        * MGL and ``latent=False``: solve on a grid of :math:`(\lambda_1, \lambda_2)` values. Choose the grid point where the eBIC is minimal.
        * MGL and ``latent=True``: in a first stage, solve SGL on a :math:`(\lambda_1, \mu_1)` for each instance :math:`k=1,\dots,K` independently. Then, do a grid search on :math:`(\lambda_1, \lambda_2)` values and for each :math:`\lambda_1` and each instance :math:`k=1,\dots,K` pick the :math:`\mu_1` value which had minimal eBIC in stage one. Then, pick again the grid point with minimal eBIC.
        
        Parameters
        ----------
        modelselect_params : dict, optional
            Dictionary with (a subset of) parameters for the grid search. This allows you to specify the grid which is used.
            Calls ``self.set_modelselect_params()``, see doc of this method for details.
        method : str, optional
            Method for choosing the best solution in the grid. 
            Options are 'AIC' (Akaike Information criterion) and 'eBIC' (extended Bayesia information criterion).
            The default is 'eBIC'.
        gamma : float, optional
            Gamma value for eBIC. Should be between 0 and 1. The larger gamma, the more eBIC tends to pick sparse solutions. 
            The default is 0.1.
        tol : float, positive, optional
            Tolerance for the primal residual used for the solver at each grid point. The default is 1e-7.
        rtol : float, positive, optional
            Tolerance for the dual residual used for the solver at each grid point. The default is 1e-7.
        
        
        Returns
        -------
        Sets ``self.reg_params`` to the best regularization parameter. Diagnostics can be accessed in ``self.modelselect_stats``.

        """
        
        assert (gamma >= 0) and (gamma <= 1), "gamma needs to be chosen as a parameter in [0,1]."
        assert method in ['eBIC', 'AIC'], "Supported evaluation methods are eBIC and AIC."
        
        self.set_modelselect_params(modelselect_params)
        
        if not np.all(self.modelselect_params['lambda1_range'] == np.sort(self.modelselect_params['lambda1_range'])[::-1]):
            print("NOTE: Ideally the lambda1 range is sorted in descending order, so the grid search is performed from sparse to dense.")
        
        ###############################
        # SINGLE GL --> GRID SEARCH lambda1/mu
        ###############################
        if not self.multiple:
            sol, all_estimates, _, stats = single_grid_search(S = self.S, lambda_range = self.modelselect_params['lambda1_range'], N = self.N, \
                               method = method, gamma = gamma, latent = self.latent, mu_range = self.modelselect_params['mu1_range'],
                               use_block = True, tol = tol, rtol = rtol)
            
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
                                                                  gamma = gamma, latent = self.latent, mu_range = self.modelselect_params['mu1_range'],
                                                                  use_block = True, tol = tol, rtol= rtol)            
                
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
    
            stats, best_ix, sol = grid_search(solver, S = self.S, N = self.N, p = self.p, reg = self.reg, l1 = self.modelselect_params['lambda1_range'], \
                                        l2 = self.modelselect_params['lambda2_range'], w2 = None, method= method, gamma = gamma, \
                                        G = self.G, latent = self.latent, mu_range = self.modelselect_params['mu1_range'], ix_mu = ix_mu, \
                                        tol = tol, rtol = rtol, verbose = False)
            
            # update the lambda1/lambda2 regularization parameters to the best grid point
            self.set_reg_params(stats['BEST'])
            # update the mu1 parameters accordingly
            if self.latent:
                # best_ix is index of best (lambda2,lambda1), we need the column of the L1 grid
                best_mu = self.modelselect_params['mu1_range'][ix_mu[:,best_ix[1]]]
                self.set_reg_params({'mu1': best_mu})
        
        ###############################
        # SET SOLUTION AND STORE INFOS
        ###############################
            
        # rescale
        if self.do_scaling:
            sol['Theta'] = self._rescale_to_covariances(sol['Theta'], self._scale)
            if self.latent:
                sol['L'] = self._rescale_to_covariances(sol['L'], self._scale)
            
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
    """
        Estimator object for the solution to the Graphical Lasso problems. 
        Reading as well the documentation of ``glasso_problem`` is highly recommended.
        Attribute naming is inspired by scikit-learn.
        
        Important attributes:
            
            * ``self.precision_``: The estimator for the sparse component of the precision matrix.
            * ``self.lowrank_``: Only relevant if ``latent=True``. The estimator for the low-rank component of the precision matrix.
            * ``self.sample_covariance_``: Empirical covariance matrix used as input for Graphical Lasso.
            
                
        Parameters
        ----------
        S : 2d/3d-array or dict
            Empirical covariance matrices.
        N : int or integer array of length K.
            Number of samples for each instance k=1,..,K.
        p : int or array of integers
            Dimension of the problem (i.e. number of variables). For non-confoming MGL, specify an array of length K.
        K : int
            Number of instances. For SGL, use K=1.
        multiple : boolean, optional
            Indicates whether SGL or MGL problem is solved. 
        latent : boolean, optional
            Indicates whether latent variables are modeled. 
        conforming : boolean, optional
            Indicates whether dimensions of MGL problem are conforming. If ``False``, then all attributes are dictionaries with keys 1,..,K. 

        """
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
        """
        calculates the eBIC for a given value of :math:`\gamma`. Note that this can differ from eBIC values in model selection because of the scaling.
        """
        if self.multiple:
            self.ebic_ = ebic(self.sample_covariance_, self.precision_, self.n_samples, gamma = gamma)          
        else:
            self.ebic_ = ebic_single(self.sample_covariance_, self.precision_, self.n_samples, gamma = gamma)        
        
        return self.ebic_
    
    def calc_adjacency(self, t = 1e-8):
        
        if self.conforming:
            self.adjacency_ = adjacency_matrix(S = self.precision_, t = t)
        
        else:
            self.adjacency_ = dict()
            for k in range(self.K):
                self.adjacency_[k] = adjacency_matrix(S = self.precision_[k], t = t)
            
        return 
        






