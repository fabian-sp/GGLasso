import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.linalg import block_diag

from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL

def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

#%%
p = 100
K = 1
N = 1000
M = 2

reg = 'GGL'


Sigma, Theta = group_power_network(p, K, M)
S, samples = sample_covariance_matrix(Sigma, N)

S = S.squeeze()


lambda1 = 0.1
Omega_0 = np.eye(p)

full_sol,_ = ADMM_SGL(S, lambda1, Omega_0, eps_admm = 1e-7, verbose = True)

sol1 = full_sol['Theta']

def get_connected_components(S, lambda1):
    
    A = (np.abs(S) > lambda1).astype(int)
    np.fill_diagonal(A, 1)
    
    numC, labelsC = connected_components(A, directed=False, return_labels=True)
    
    allC = list()
    for i in range(numC):
        # need hstack for avoiding redundant dimensions
        thisC = np.hstack(np.argwhere(labelsC == i))
        
        allC.append(thisC)
    
    return numC, allC
    
#%%
numC, allC =  get_connected_components(S, lambda1)

allSOL = list()

for i in range(numC):
    C = allC[i]
    
    # single node connected components have a closed form solution, see Witten, Friedman, Simon "NEW INSIGHTS FOR THE GRAPHICAL LASSO "
    if len(C) == 1:
        # we use the OFF-DIAGONAL l1-penalty, otherwise it would be 1/(S[C,C]+lambda1)
        closed_sol = 1/(S[C,C])
        allSOL.append(closed_sol)
        
    # else solve Graphical Lasso for the corresponding block       
    else:
        block_S = S[np.ix_(C,C)]
        block_sol,_ =  ADMM_SGL(S = block_S, lambda1 = lambda1, eps_admm = 1e-7, Omega_0 = Omega_0[np.ix_(C,C)], verbose = True)
        
        allSOL.append(block_sol['Theta'])
    
# stack together all blocks, but still the indices are permuted    
sol_permuted = block_diag(*allSOL)

# compute inverse permutation
per = np.hstack(allC)
per1 = invert_permutation(per)

# apply
sol2 = sol_permuted[np.ix_(per1,per1)]
    
# check
np.linalg.norm(sol2-sol1)/ np.linalg.norm(sol1)



#def block_SGL(S, lambda1, Omega_0, rho=1., max_iter = 1000, tol = 1e-5 , verbose = False, measure = False):
    
    
    
#import networkx as nx
#G = nx.from_numpy_array(A)
#nx.draw_spring(G, with_labels = True)
