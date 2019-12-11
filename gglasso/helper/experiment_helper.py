import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import seaborn as sns
import networkx as nx

from .basic_linalg import Sdot


def lambda_parametrizer(l2 = 0.05, w2 = 0.5):
    
    w1 = l2/(w2*np.sqrt(2))
    l1 = w1 - l2/np.sqrt(2)
    return l1


def lambda_grid(num1 = 5, num2 = 2, reg = 'GGL'):
    """
    num1: number of grid point for lambda 1
    num2: number of grid point for lambda 2
    reg: grid for GGL or FGL (interpretation changes)
    creates a grid of lambda 1 lambda 1 values
    idea: the grid goes from smaller to higher values when going down/right
    """
    l2 = np.logspace(start = -2, stop = -1, num = num2, base = 10)
    
    if reg == 'GGL':
        w2 = np.linspace(0.2, 0.8, num1)
        l2grid, w2grid = np.meshgrid(l2,w2)
        L1 = lambda_parametrizer(l2grid, w2grid)
        L2 = l2grid.copy()
    elif reg == 'FGL':
        l1 = np.logspace(start = -2, stop = -.5, num = num1, base = 10)
        L2, L1 = np.meshgrid(l2,l1)
        
    return L1.squeeze(), L2.squeeze()
           
def adjacency_matrix(S , t = 1e-9):
    A = (np.abs(S) >= t).astype(int)
    # do not count diagonal entries as edges
    if len(S.shape) == 3:
        for k in np.arange(S.shape[0]):
            np.fill_diagonal(A[k,:,:], 0)
    else:
        np.fill_diagonal(A, 0)
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
    return np.linalg.norm(S_sol - S_true)/np.linalg.norm(S_true)

def aic(S,Theta, N):
    (K,p,p) = S.shape
    # counts the non-zero off-diagonal elements
    nonzero_count = (abs(Theta) >= 1e-3).sum(axis=(1,2)) - p
    aic = 0
    for k in np.arange(K):
        aic += N*Sdot(S[k,:,:], Theta[k,:,:]) - N*np.log(np.linalg.det(Theta[k,:,:])) + 1*nonzero_count[k]
        
    return aic

def l1norm_od(Theta):
    """
    calculates the off-diagonal l1-norm of a matrix
    """
    (p1,p2) = Theta.shape
    res = 0
    for i in np.arange(p1):
        for j in np.arange(p2):
            if i == j:
                continue
            else:
                res += abs(Theta[i,j])
                
    return res

def deviation(Theta):
    """
    calculates the deviation of subsequent Theta estimates
    deviation = off-diagonal l1 norm
    """
    #tmp = np.roll(Theta, 1, axis = 0)
    (K,p,p) = Theta.shape
    d = np.zeros(K-1)
    for k in np.arange(K-1):
        d[k] = l1norm_od(Theta[k+1,:,:] - Theta[k,:,:])
        
    return d

#################################################################################################################
############################ DRAWING FUNCTIONS ################################################################################
#################################################################################################################

def get_default_plot_aes():
    plot_aes = {'marker' : 'o', 'linestyle' : '-', 'markersize' : 5}
    
    return plot_aes


def get_default_color_coding():
    mypal = sns.color_palette("Set2")
    
    color_dict = {}    
    color_dict['truth'] = mypal[0]
    color_dict['GLASSO'] = mypal[1]
    color_dict['ADMM'] = mypal[2]
    color_dict['PPDNA'] = mypal[3]

    return color_dict

def get_graph_aes(with_edge_col = True):
    aes = {'node_size' : 100, 'node_color' : 'lightslategrey', 'edge_color' : 'lightslategrey', 'width' : 1.5}
    
    if not with_edge_col:
        del aes['edge_color']
    return aes
                
def draw_group_graph(Theta , t = 1e-9):
    """
    Draws a network with Theta as precision matrix
    """
    
    assert len(Theta.shape) == 3
    (K,p,p) = Theta.shape
    A = adjacency_matrix(Theta , t)
    
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

def draw_group_heatmap(Theta, ax = None, t = 1e-9):
    (K,p,p) = Theta.shape
    A = adjacency_matrix(Theta, t)
    mask = A.sum(axis=0) == 0
    
    if ax == None:
        fig,ax = plt.subplots(nrows = 1, ncols = 1)
    with sns.axes_style("white"):
        sns.heatmap(A.sum(axis=0), mask = mask, ax = ax, square = True, cmap = 'Blues', vmin = 0, vmax = K, linewidths=.5, cbar_kws={"shrink": .5})

def plot_block_evolution(ax, start, stop, Theta, method, color_dict):
    (K,p,p) = Theta.shape
    plot_aes = get_default_plot_aes()
    for i in np.arange(start, stop):
        for j in np.arange(start = i+1, stop = stop):
            
            x = np.arange(K)
            ax.plot(x, abs(Theta[:,i,j]), c=color_dict[method], label = method if (i == start) & (j == start+1) else "", **plot_aes)
    
    ax.legend(labels = [method])   
    ax.set_ylim(0,0.5)
    return

def plot_evolution(results, block = None, L = None, start = None, stop = None):
    """
    plots the evolution of edges for block
    alternatively specify start and stop index of the matrix
    """
    color_dict = get_default_color_coding()
    
    if block is not None:
        assert L is not None
        
    if block is not None:
        start = block*L
        stop = (block+1)*L
    
    with sns.axes_style("whitegrid"):
        fig,axs = plt.subplots(nrows = 2, ncols = 2)
        plot_block_evolution(axs[0,0], start, stop, results.get('truth').get('Theta'), 'truth', color_dict)
        plot_block_evolution(axs[0,1], start, stop, results.get('PPDNA').get('Theta'), 'PPDNA', color_dict)
        plot_block_evolution(axs[1,0], start, stop, results.get('ADMM').get('Theta'), 'ADMM', color_dict)
        plot_block_evolution(axs[1,1], start, stop, results.get('GLASSO').get('Theta'), 'GLASSO', color_dict)
    
    fig.suptitle('Precision matrix entries - evolution over time')
    return

def plot_deviation(results):
    """
    plots the temporal deviation
    """
    color_dict = get_default_color_coding()
    plot_aesthetics = get_default_plot_aes()
    
    with sns.axes_style("whitegrid"):
        fig,ax = plt.subplots(nrows=1,ncols=1)
    
        for m in list(results.keys()):
            d = deviation(results.get(m).get('Theta'))
            ax.plot(d, c = color_dict[m], **plot_aesthetics)
                
        ax.set_ylabel('Temporal Deviation')
        ax.set_xlabel('Time (k=1,...,K)')
        ax.legend(labels = list(results.keys()))

    return

#################################################################################################################
############################ GIF ################################################################################
#################################################################################################################

def plot_single_heatmap(k, Theta, method, ax):
    """
    plots a heatmap of the adjacency matrix at index k
    """
    A = adjacency_matrix(Theta[k,:,:])
    mask = (A == 0) 
    
    #with sns.axes_style("white"):
    col = get_default_color_coding()[method]
    this_cmap = sns.light_palette(col, as_cmap=True)
    
    ax.cla()
    sns.heatmap(A, mask = mask, ax = ax, square = True, cmap = this_cmap, vmin = 0, vmax = 1, linewidths=.5, cbar = False)
    ax.set_title(f"Precision matrix at timestamp {k}")
    
    return 

def single_heatmap_animation(Theta, method = 'truth', save = False):
    
    (K,p,p) = Theta.shape
    
    fig, ax = plt.subplots(1,1)
    fargs = (Theta, method, ax,)
    
    def init():
        ax.cla()
        A = np.zeros((p, p))
        mask = (A == 0) 
        sns.heatmap(A, mask = mask, ax = ax, square = True, cmap = 'Blues', vmin = 0, vmax = 1, linewidths=.5, cbar = False)

    anim = FuncAnimation(fig, plot_single_heatmap, frames = K, init_func=init, interval= 1000, fargs = fargs, repeat = True)
    
    if save:    
        anim.save("single_network.gif", writer='imagemagick')
        
    return anim


def plot_multiple_heatmap(k, Theta, results, axs):
    
    plot_single_heatmap(k, Theta, 'truth', axs[0,0])
    plot_single_heatmap(k, results.get('PPDNA').get('Theta'), 'PPDNA', axs[0,1])
    plot_single_heatmap(k, results.get('ADMM').get('Theta'), 'ADMM', axs[1,0])
    plot_single_heatmap(k, results.get('GLASSO').get('Theta'), 'GLASSO', axs[1,1])
    
    return

def multiple_heatmap_animation(Theta, results, save = False):
    (K,p,p) = Theta.shape
    fig, axs = plt.subplots(nrows = 2, ncols=2)

    def init():
        for ax in axs.ravel():
            ax.cla()
        A = np.zeros((p, p))
        mask = (A == 0) 
        for ax in axs.ravel():
            sns.heatmap(A, mask = mask, ax = ax, square = True, cmap = 'Blues', vmin = 0, vmax = 1, linewidths=.5, cbar = False)


    fargs = (Theta, results, axs,)
    anim = FuncAnimation(fig, plot_multiple_heatmap, frames = K, init_func=init, interval= 1000, fargs = fargs, repeat = True)
       
    if save:    
        anim.save("multiple_network.gif", writer='imagemagick')
    
    return anim    

#################################################################################################################
