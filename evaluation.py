import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


edge_ths = 1e-3

# A is the adjacency matrix
A = (Sigma_inv >= edge_ths).astype(int)


G = nx.from_numpy_array(A)


aes = {'node_size' : 100,
 'node_color' : 'steelblue',
 'edge_color' : 'lightslategrey',
 'width' : 1.5}

plt.figure()
nx.draw_kamada_kawai(G, **aes)



def discovery_rate(S_sol , S_true, t):
    A_sol = (np.abs(S_sol) >= t).astype(int)
    A_true = (np.abs(S_true) >= t).astype(int)
    
    
    true_edges = A_true.sum(axis = (1,2))
    positive_edges = A_sol.sum(axis = (1,2))
    
    # true positive edge ratio
    tp = (A_sol + A_true == 2).sum(axis = (1,2)) / true_edges
    
    # false positive edge ratio
    fp = (A_true - A_sol == -1).sum(axis = (1,2)) / positive_edges
    
    # true negative edge ration
    nd =  (A_true - A_sol == 1).sum(axis = (1,2)) / true_edges
    
    return tp, fp, nd, true_edges, positive_edges
    
    
    
    
    
    
    