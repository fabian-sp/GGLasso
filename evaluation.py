import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def get_graph_aes(with_edge_col = True):
    aes = {'node_size' : 100, 'node_color' : 'steelblue', 'edge_color' : 'lightslategrey', 'width' : 1.5}
    
    if not with_edge_col:
        del aes['edge_color']
           
    return aes

def adjacency_matrix(S , t = 1e-9):
    return (np.abs(S) >= t).astype(int)


def discovery_rate(S_sol , S_true, t = 1e-9):
    A_sol = adjacency_matrix(S_sol, t)
    A_true = adjacency_matrix(S_true, t)
    
    
    true_edges = A_true.sum(axis = (1,2))
    positive_edges = A_sol.sum(axis = (1,2))
    
    # true positive edge ratio
    tp = (A_sol + A_true == 2).sum(axis = (1,2)) / true_edges
    
    # false positive edge ratio
    fp = (A_true - A_sol == -1).sum(axis = (1,2)) / positive_edges
    
    # true negative edge ration
    nd =  (A_true - A_sol == 1).sum(axis = (1,2)) / true_edges
    
    return tp, fp, nd, true_edges, positive_edges


def draw_group_graph(Omega , t = 1e-9):
    
    assert len(Omega.shape) == 3
    (K,p,p) = Omega.shape
    A = adjacency_matrix(Omega , t)
    
    gA = A.sum(axis=0)
    
    G = nx.from_numpy_array(gA)
    
    aes = get_graph_aes(False)
    
    edge_col = []
    for e in G.edges:
        edge_col.append( gA[e[0], e[1]])
        
    fig = plt.figure()
    nx.draw_kamada_kawai(G, with_labels = True, edge_color = edge_col, edge_cmap = plt.cm.RdYlGn, edge_vmin = 0, edge_vmax = K, **aes)
    
    return fig
    
    
    
    
    
    

    
    
    
    
    
    
    