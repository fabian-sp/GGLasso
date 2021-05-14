"""
author: Fabian Schaipp
"""
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from gglasso.helper.utils import hamming_distance, sparsity, mean_sparsity, deviation, get_K_identity
from gglasso.helper.utils import zero_replacement, normalize, log_transform

from gglasso.helper.basic_linalg import scale_array_by_diagonal
from gglasso.helper.ext_admm_helper import get_K_identity as id_dict
from gglasso.helper.ext_admm_helper import construct_indexer, create_group_array, check_G, consensus
from gglasso.helper.model_selection import lambda_parametrizer, map_l_to_w


def test_lambda_w_map():
    l1 = 0.1
    l2 = 0.05
    w1, w2 = map_l_to_w(l1, l2)
    l2_ = lambda_parametrizer(l1, w2)

    assert l2 == l2_
    return 

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

def test_hamming():
    p = 20
    A = np.random.rand(p,p)
    B = np.random.rand(p,p)
    
    assert hamming_distance(A,B, t= 1e-2) == hamming_distance(B,A, t=1e-2)
    assert hamming_distance(A,B, t=1e-2) <= p**2-p
    assert hamming_distance(A,B, t = 1e-10) <= hamming_distance(A,B, t = 1e-2)
    assert hamming_distance(np.eye(p), np.ones((p,p))) == p**2-p

    return 

def test_scale_by_diagonal():
    p = 20
    A = np.random.rand(p,p)
    d = np.diag(A)
    A1 = scale_array_by_diagonal(A, d = None)
    A2 = scale_array_by_diagonal(A1, d = 1/d)

    assert_array_almost_equal(A, A2, decimal = 10)
    return 


def test_create_G():
    """
    test for creating the bookeeping array G
    """
    K = 5
    p = 50
    N = 100
    sub_p = [40, 35, 42, 23, 47]
    
    # create dummy data
    all_data = list()
    for k in np.arange(K):
        S = np.sort(np.random.choice(a = np.arange(p), size = sub_p[k], replace = False))
        dummy_vals = np.random.rand(sub_p[k], N)
        all_data.append(pd.DataFrame(dummy_vals, index = S, columns = np.arange(N)))
    
    # test 
    ix_exist, ix_location = construct_indexer(all_data)
    
    assert np.all(ix_exist.sum(axis=0) == sub_p)
    
    G = create_group_array(ix_exist, ix_location, min_inst = 2)
    check_G(G, p)
    
    # now check consensus function
    # create solution of ones --> every edge is present
    sol = dict()
    for k in np.arange(K):
        sol[k] = np.ones((sub_p[k],sub_p[k]))
    
    
    group_size = (G[0,:,:] != -1).sum(axis = 1)
    nnz, _, _ = consensus(sol, G)
    
    assert np.all(nnz == group_size)

    return
    
def test_clr_functions():
    N = 100
    p = 200
    X = np.random.randint(0,100,(p,N))
    
    # create some zeros
    X[X%4==1] = 0
    
    X = pd.DataFrame(X)
    
    # after clr trasnform, 
    X = zero_replacement(X)
    X = normalize(X)
    
    # after normalizing each sample is a point on the simplex
    assert_array_almost_equal(X.sum(axis=0).values, np.ones(N, float))
    
    # after log transform, sum of components = 0 for each sample
    X = log_transform(X)
    assert_array_almost_equal(X.sum(axis=0).values, np.zeros(N, float))
    
    return
        
        
        
        