import numpy as np
from gglasso.helper.basic_linalg import trp

assert_tol = 1e-5

class glasso_problem:
    
    def __init__(self, S, reg, reg_params, latent = False):
        
        self.S = S
        
        self.latent = latent
        self.reg = reg
        self.reg_params = reg_params
        
        
        
        
    def derive_problem_formulation(self):
        
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
                
                
        elif type(self.S) == dict:
            
            assert len(self.S.keys() > 1), "Covariance data is a dictionary with only one key. This is a Single Graphical Lasso problem. Specify S as 2d-array."
            self.conforming = False
            self.multiple = True
            self.check_covariance_dict()
    
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
    
    def check_covariance_dict(self):
        self.K = len(self.S.keys())
        
        #TODO: assert for keys being equal to 1..K
        
        self.p = np.zeros(self.K, dtype = int)
        
        for k in range(self.K):
            assert self.S[k].shape[0] == self.S[k].shape[1], f"Dimensions are not correct, 1st and 2nd dimension have to match but do not match for instance {k}."
            assert np.max(np.abs(self.S[k] - self.S[k].T)) <= assert_tol, f"Covariance data for instance {k} is not symmetric."
            
            self.p[k]= self.S[k].shape[0]
            
        return
    
    
    