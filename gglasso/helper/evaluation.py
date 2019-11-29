import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def get_graph_aes(with_edge_col = True):
    aes = {'node_size' : 100, 'node_color' : 'lightslategrey', 'edge_color' : 'lightslategrey', 'width' : 1.5}
    
    if not with_edge_col:
        del aes['edge_color']
           
    return aes

def adjacency_matrix(S , t = 1e-9):
    A = (np.abs(S) >= t).astype(int)
    # do not count diagonal entries as edges
    if len(S.shape) == 3:
        for k in np.arange(S.shape[0]):
            np.fill_diagonal(A[k,:,:], 0)
    return A


def discovery_rate(S_sol , S_true, t = 1e-9):
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
    
    
    
    
    
    

    
    
    
    
    
    
    