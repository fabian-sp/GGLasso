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

