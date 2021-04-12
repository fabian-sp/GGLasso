"""
author: Fabian Schaipp
"""
import numpy as np
from gglasso.helper.utils import hamming_distance, sparsity, mean_sparsity, deviation, get_K_identity

from gglasso.helper.ext_admm_helper import get_K_identity as id_dict

def test_sparsity_id():
    
    I = np.eye(100)
    assert sparsity(I) == 0
    
    return

def test_sparsity_K_id():
    
    I = get_K_identity(10, 100)
    assert mean_sparsity(I) == 0
    
    return

def test_sparsity_K_id_dict():
    
    I = id_dict(100*np.ones(5, dtype = int))
    assert mean_sparsity(I) == 0
    
    return

def test_deviation_K_id():
    
    I = np.ones((5,10,10))
    assert np.all(deviation(I) == 0)
    
    return



