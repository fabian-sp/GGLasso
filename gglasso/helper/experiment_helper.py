import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .basic_linalg import Sdot

def lambda_parametrizer(l2 = 0.05, w2 = 0.5):
    
    w1 = l2/(w2*np.sqrt(2))
    l1 = w1 - l2/np.sqrt(2)
    return l1


def lambda_grid(num1 = 5, num2 = 2):
    l2 = np.logspace(start = -1, stop = -2, num = num2, base = 10)
    #w2 = np.linspace(0.1, 0.7, num = num1)
    #l1 = lambda_parametrizer(l2, w2)
    
    l1 = np.logspace(start = -.5, stop = -2, num = num1, base = 10)
    
    L1, L2 = np.meshgrid(l1,l2)
    return L1.T, L2.T
           
def adjacency_matrix(S , t = 1e-9):
    A = (np.abs(S) >= t).astype(int)
    # do not count diagonal entries as edges
    if len(S.shape) == 3:
        for k in np.arange(S.shape[0]):
            np.fill_diagonal(A[k,:,:], 0)
    return A


def discovery_rate(S_sol , S_true, t = 1e-9):
    if len(S_true.shape) == 2:
        print("Warning: function designed for 3-dim arrays")
        S_true = S_true[np.newaxis,:,:]
        S_sol = S_sol[np.newaxis,:,:]
        
    (K,p,p) = S_true.shape
    
    A_sol = adjacency_matrix(S_sol, t)
    A_true = adjacency_matrix(S_true, t)
        
    true_edges = A_true.sum(axis = (1,2))
    true_non_edges =  p*(p-1) - true_edges
    positive_edges = A_sol.sum(axis = (1,2))
    
    # true positive edge ratio
    tp = (A_sol + A_true == 2).sum(axis = (1,2)) / true_edges 
    # false positive edge ratio
    fp = (A_true - A_sol == -1).sum(axis = (1,2)) / true_non_edges 
    # true negative edge ratio
    nd =  (A_true - A_sol == 1).sum(axis = (1,2)) / true_edges
    
    res = {'TPR': tp.mean(), 'FPR' : fp.mean(), 'TNR' : nd.mean(), 'TE' : true_edges}
    
    return res

def error(S_sol , S_true):
    return np.linalg.norm(S_sol - S_true)

def aic(S,Theta, N):
    (K,p,p) = S.shape
    aic = 0
    for k in np.arange(K):
        aic += N*( Sdot(S[k,:,:], Theta[k,:,:]) - np.log(np.linalg.det(Theta[k,:,:])) ) + 2*(abs(Theta[k,:,:]) >= 1e-4).sum()
        
    return aic


# Drawing functions


def get_graph_aes(with_edge_col = True):
    aes = {'node_size' : 100, 'node_color' : 'lightslategrey', 'edge_color' : 'lightslategrey', 'width' : 1.5}
    
    if not with_edge_col:
        del aes['edge_color']
    return aes
        
        
def draw_group_graph(Omega , t = 1e-9):
    """
    Draws a network with Omega as precision matrix
    """
    
    assert len(Omega.shape) == 3
    (K,p,p) = Omega.shape
    A = adjacency_matrix(Omega , t)
    
    gA = A.sum(axis=0)
    G = nx.from_numpy_array(gA)
    aes = get_graph_aes(with_edge_col = False)
    
    edge_col = []
    for e in G.edges:
        edge_col.append( gA[e[0], e[1]])
    
    
    fig = plt.figure()
    #nx.draw_shell(G, with_labels = True, edge_color = edge_col, edge_cmap = plt.cm.RdYlGn, edge_vmin = 0, edge_vmax = K, **aes)
    nx.draw_spring(G, with_labels = True, edge_color = edge_col, edge_cmap = plt.cm.RdYlGn, edge_vmin = 0, edge_vmax = K, **aes)
    
    return fig

