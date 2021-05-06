"""
author: Fabian Schaipp
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.ticker

import seaborn as sns
import networkx as nx

from .basic_linalg import adjacency_matrix
from .utils import deviation

#################################################################################################################
############################ FOR EXPERIMENT SETUP ##################################################################
#################################################################################################################

def lambda_parametrizer(l2 = 0.05, w2 = 0.5):
    
    w1 = l2/(w2*np.sqrt(2))
    l1 = w1 - l2/np.sqrt(2)
    return l1


def lambda_grid(num1 = 5, num2 = 2, reg = 'GGL'):
    """
    num1: number of grid point for lambda 1
    num2: number of grid point for lambda 2
    reg: grid for GGL or FGL (interpretation changes)
    creates a grid of lambda1 / lambda2 values
    idea: the grid goes from smaller to higher values when going down/right
    """
    
    if reg == 'GGL':
        l2 = np.logspace(start = -0.5, stop = -2.5, num = num2, base = 10)
        w2 = np.linspace(0.2, 0.5, num1)
        l2grid, w2grid = np.meshgrid(l2,w2)
        L1 = lambda_parametrizer(l2grid, w2grid)
        L2 = l2grid.copy()
    elif reg == 'FGL':
        l2 = 2*np.logspace(start = -1, stop = -3, num = num2, base = 10)
        l1 = 2*np.logspace(start = -1, stop = -3, num = num1, base = 10)
        L2, L1 = np.meshgrid(l2,l1)
        w2 = None
        
    return L1.squeeze(), L2.squeeze(), w2
           
        
def discovery_rate(S_sol , S_true, t = 1e-10):
    if len(S_true.shape) == 2:
        print("Warning: function designed for 3-dim arrays")
        S_true = S_true[np.newaxis,:,:]
        S_sol = S_sol[np.newaxis,:,:]
        
    (K,p,p) = S_true.shape
    
    A_sol = adjacency_matrix(S_sol, t)
    A_true = adjacency_matrix(S_true, t)
        
    true_edges = A_true.sum(axis = (1,2))
    true_non_edges =  p*(p-1) - true_edges
    sparsity = A_sol.sum(axis = (1,2))/(p**2-p)
    
    # true positive edge ratio
    tp = (A_sol + A_true == 2).sum(axis = (1,2)) / true_edges 
    # false positive edge ratio
    fp = (A_true - A_sol == -1).sum(axis = (1,2)) / true_non_edges 
    # true negative edge ratio
    nd =  (A_true - A_sol == 1).sum(axis = (1,2)) / true_edges
    
    # differential edges
    diff_true = (A_true.sum(axis = 0) < K).astype(int) * (A_true.sum(axis = 0) >= 1).astype(int)
    diff_sol = (A_sol.sum(axis = 0) < K).astype(int) * (A_sol.sum(axis = 0) >= 1).astype(int)
    
    tp_diff = (diff_sol + diff_true == 2).sum() #/ diff_true.sum() 
    # divide by true npn-differential edges
    fp_diff = (diff_sol - diff_true == 1).sum() #/ (p*(p-1) - diff_true.sum())
    
    res = {'TPR': tp.mean(), 'FPR' : fp.mean(), 'TNR' : nd.mean(), 'SP' : sparsity.mean(), 'TPR_DIFF' : tp_diff, 'FPR_DIFF' : fp_diff}
    
    return res

def error(S_sol , S_true):
    return np.linalg.norm(S_sol - S_true)/np.linalg.norm(S_true)


#################################################################################################################
############################ DRAWING FUNCTIONS ##################################################################
#################################################################################################################

path_ggl_powerlaw = '../plots/ggl_powerlaw/'
path_fgl_powerlaw = '../plots/fgl_powerlaw/'
path_ggl_runtime = '../plots/ggl_runtime/' 

default_size_big = (10,7)
default_size_small = (6,5)

def get_default_plot_aes():
    plot_aes = {'marker' : 'o', 'markersize' : 4}
    
    return plot_aes


def get_default_color_coding():
    mypal = sns.color_palette("Dark2", 10)
    
    color_dict = {}    
    color_dict['truth'] = sns.color_palette("YlGnBu", 10)[-1] #'darkblue'
    color_dict['SGL'] = mypal[7]
    color_dict['ADMM'] = mypal[0]
    color_dict['PPDNA'] = mypal[1]
    color_dict['LTGL'] = mypal[5]

    return color_dict


def draw_group_heatmap(Theta, method = 'truth', ax = None, t = 1e-5, save = False):
    (K,p,p) = Theta.shape
    A = adjacency_matrix(Theta, t)
    mask = A.sum(axis=0) == 0
    
    col = get_default_color_coding()[method]
    this_cmap = sns.light_palette(col, as_cmap=True)
    
    
    if ax == None:
        fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = default_size_big)
        fig.suptitle("Distribution of edges (color indicates number of instances with edge present)")
    with sns.axes_style("white"):
        sns.heatmap(A.sum(axis=0), mask = mask, ax = ax, square = True, cmap = this_cmap, vmin = 0, vmax = K, linewidths=.5, cbar_kws={"shrink": .5})

    if save:
        fig.savefig(path_ggl_powerlaw + method +'_heatmap.pdf')
    
    
    return
        
def plot_block_evolution(ax, start, stop, Theta, method, color_dict):
    (K,p,p) = Theta.shape
    plot_aes = get_default_plot_aes()
    for i in np.arange(start, stop):
        for j in np.arange(start = i+1, stop = stop):
            
            x = np.arange(K) + 1
            ax.plot(x, abs(Theta[:,i,j]), c=color_dict[method], label = method if (i == start) & (j == start+1) else "", **plot_aes)
    
    ax.legend(labels = [method])   
    ax.set_ylim(0,0.35)
    return

def plot_evolution(results, block = None, L = None, start = None, stop = None, save = False):
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
        fig,axs = plt.subplots(nrows = 2, ncols = 2, figsize = default_size_small)
        plot_block_evolution(axs[0,0], start, stop, results.get('truth').get('Theta'), 'truth', color_dict)
        plot_block_evolution(axs[0,1], start, stop, results.get('ADMM').get('Theta'), 'ADMM', color_dict)
        plot_block_evolution(axs[1,0], start, stop, results.get('LTGL').get('Theta'), 'LTGL', color_dict)
        plot_block_evolution(axs[1,1], start, stop, results.get('SGL').get('Theta'), 'SGL', color_dict)
    
    fig.suptitle('Precision matrix entries - evolution over time')
    
    if save:
        fig.savefig(path_fgl_powerlaw + 'block_' + str(block) + '_evolution.pdf')
    return

def plot_deviation(results, latent = None, save = False):
    """
    plots the temporal deviation
    """
    color_dict = get_default_color_coding()
    plot_aesthetics = get_default_plot_aes()
    
    with sns.axes_style("whitegrid"):
        fig,ax = plt.subplots(nrows=1,ncols=1, figsize = default_size_big)
    
        for m in list(results.keys()):
            d = deviation(results.get(m).get('Theta'))
            if m in ['truth', 'SGL']:
                ls = '--'
            else:
                ls = '-'
            ax.plot(d, c = color_dict[m], linestyle = ls, **plot_aesthetics)
                
        ax.set_ylabel('Temporal Deviation Theta')
        ax.set_xlabel('Time (k=1,...,K)')
        labels = list(results.keys())
        ax.legend(labels = labels, loc = 'upper left')
        if latent is not None:
            ax2 = ax.twinx()
            ax2.plot(deviation(latent), linestyle = '--', **plot_aesthetics)
            ax2.set_ylabel('Temporal Deviation Latent variables')
            ax2.legend(labels = ['Latent variables'], loc = 'upper right')
        
    if save:
        fig.savefig(path_fgl_powerlaw + 'deviation.pdf')

    return

def plot_runtime(iA, iP, vecN, save = False):
    """
    plots runtime and KKT residual for PPDNA and ADMM method
    """
    color_dict = get_default_color_coding()
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (11,7)) 
    
    for j in np.arange(len(vecN)):       
        ax = axs.reshape(-1)[j]
        with sns.axes_style("whitegrid"):
            
            p1 = ax.plot(iA[j]['residual'], c = color_dict['ADMM'], label = 'ADMM residual')
            p2 = ax.plot(iP[j]['residual'], c = color_dict['PPDNA'], marker = 'o', markersize = 3, label = 'PPDNA residual')
            
            #ax.tick_params(axis='both', which='major', labelsize=7)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylim(1e-6,0.2)
            
            ax2 = ax.twinx()
            ax2.set_xscale('log')
            p3 = ax2.plot(iA[j]['runtime'].cumsum(), linestyle = '--', c = color_dict['ADMM'], alpha = 0.7, label = 'ADMM runtime')
            p4 = ax2.plot(iP[j]['runtime'].cumsum(), linestyle = '--', c = color_dict['PPDNA'], marker = 'o', markersize = 3, alpha = 0.7, label = 'PPDNA runtime')
            
            # plot start x axis at 
            ax.vlines(iP[j]['iter_admm']-1, 0, 0.2, 'grey')
            ax.set_xlim(max(iP[j]['iter_admm'] - 5,1), )
            
            #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            
            if j in [0,2]:
                ax.set_ylabel('KKT residual')
            if j in [1,3]:
                ax2.set_ylabel('Cumulated runtime [sec]')
            if j in [2,3]:
                ax.set_xlabel('Iteration number')
            
            ax.set_title(f'Sample size = {vecN[j]}')
            
            lns = p1+p2+p3+p4
            labs = [l.get_label() for l in lns]
            fig.legend(lns, labs, loc="upper right")
             
        if save:
            fig.savefig(path_ggl_runtime + 'runtimeN.pdf', dpi = 300)

def plot_fpr_tpr(FPR, TPR, ix, ix2, FPR_GL = None, TPR_GL = None, W2 = [], save = False):
    """
    plots the FPR vs. TPR pathes
    ix and ix2 are the lambda values with optimal eBIC/AIC
    """
    plot_aes = get_default_plot_aes()

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(1,1,figsize = default_size_small)
        ax.plot(FPR.T, TPR.T, **plot_aes)
        if FPR_GL is not None:
            ax.plot(FPR_GL, TPR_GL, c = 'grey', linestyle = '--', **plot_aes)
        
        ax.plot(FPR[ix], TPR[ix], marker = 'o', color = "white", fillstyle = 'none', markersize = 12, markeredgecolor = 'grey')
        ax.plot(FPR[ix2], TPR[ix2], marker = 'D', color = "white", fillstyle = 'none', markersize = 12, markeredgecolor = 'grey')
    
        ax.set_xlim(-.01,1)
        ax.set_ylim(-.01,1)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        labels = [f"w2 = {w}" for w in W2] 
        if FPR_GL is not None:
            labels.append("SGL")
            
        labels.append("eBIC")
        labels.append("AIC")    
            
        ax.legend(labels = labels, loc = 'lower right')
    
    ax.set_xlim()
    
    fig.suptitle('Discovery rate for different regularization strengths')
    if save:
        fig.savefig(path_ggl_powerlaw + 'fpr_tpr.pdf', dpi = 300)
    
    return fig, ax

def plot_diff_fpr_tpr(DFPR, DTPR, ix, ix2, DFPR_GL = None, DTPR_GL = None, W2 = [], save = False):
    """
    plots the FPR vs. TPR pathes 
    _GL indicates the solution of single Graphical Lasso
    ix and ix2 are the lambda values with optimal eBIC/AIC
    """
    plot_aes = get_default_plot_aes()
    
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(1,1, figsize = default_size_small)
        ax.plot(DFPR.T, DTPR.T, label = W2, **plot_aes)
        if DFPR_GL is not None:
            ax.plot(DFPR_GL, DTPR_GL, c = 'grey', linestyle = '--', **plot_aes)
        
        ax.plot(DFPR[ix], DTPR[ix], marker = 'o', color = "white", fillstyle = 'none', markersize = 12, markeredgecolor = 'grey')
        ax.plot(DFPR[ix2], DTPR[ix2], marker = 'D', color = "white", fillstyle = 'none', markersize = 12, markeredgecolor = 'grey')
                
        ax.set_xlim(-10,300)
        #ax.set_ylim(-.01,1)
        
        ax.set_xlabel('FP Differential Edges')
        ax.set_ylabel('TP Differential Edges')
        labels = [f"w2 = {w}" for w in W2]
        if DFPR_GL is not None:
            labels.append("SGL")
        
        labels.append("eBIC")
        labels.append("AIC")
        ax.legend(labels = labels, loc = 'lower right')
        
    fig.suptitle('Discovery of differential edges')
    if save:
        fig.savefig(path_ggl_powerlaw + 'diff_fpr_tpr.pdf', dpi = 300)
    
    return fig, ax


def plot_error_accuracy(EPS, ERR, L2, save = False):
    pal = sns.color_palette("GnBu_d", len(L2))
    plot_aes = get_default_plot_aes()
    
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(1,1,figsize = default_size_big)
        for l in np.arange(len(L2)):
            ax.plot(EPS, ERR[l,:], c=pal[l],**plot_aes )
    
        ax.set_xlim(EPS.max()*2 , EPS.min()/2)
        ax.set_ylim(0,0.3)
        ax.set_xscale('log')
        
        ax.set_xlabel('Solution accuracy')
        ax.set_ylabel('Total relative error')
        ax.legend(labels = ["l2 = " + "{:.2E}".format(l) for l in L2])
        
    fig.suptitle('Total error for different solution accuracies')
    if save:
        fig.savefig(path_ggl_powerlaw + 'error.pdf', dpi = 300)
    
    return fig, ax

def plot_gamma_influence(gammas, GTPR, GFPR, save = False):
    fig, ax = plt.subplots(1,1, figsize = default_size_big)   
    
    ax.plot(gammas, GTPR, c = 'green', label = 'TPR')
    ax2 = ax.twinx() 
    ax2.plot(gammas, GFPR, c = 'red', label = 'FPR')
    
    ax.grid(linestyle = '--') 
    ax.set_ylabel('gamma')    
    ax.set_ylabel('TPR')
    ax2.set_ylabel('FPR')

    fig.legend()
    
    if save:
        fig.savefig(path_ggl_powerlaw + 'gamma.pdf', dpi = 300)
        
    return 

        
def surface_plot(L1, L2, C, name = 'eBIC'):
    fig = plt.figure(figsize = (8,5))  
        
    if type(C) == np.ndarray:
        ax = fig.gca(projection='3d')
        single_surface_plot(L1, L2, C, ax, name = name)
             
    else:
        gammas = list(C.keys())
        
        for j in np.arange(len(gammas)):
            ax = fig.add_subplot(2, 2, j+1, projection='3d')
            single_surface_plot(L1, L2, C[gammas[j]], ax, name = name)
            if gammas is not None:
                ax.set_title(rf"$\gamma = $ {gammas[j]}")
    
    return fig

def single_surface_plot(L1, L2, C, ax, name = 'eBIC'):
    
    X = np.log10(L1)
    Y = np.log10(L2)
    Z = np.log(C)
    ax.plot_surface(X, Y, Z , cmap = plt.cm.ocean, linewidth=0, antialiased=True)
    
    ax.set_xlabel(r'$\lambda_1$', fontsize = 14)
    ax.set_ylabel(r'$\lambda_2$', fontsize = 14)
    #ax.set_xlabel(r'$w_1$', fontsize = 14)
    #ax.set_ylabel(r'$w_2$', fontsize = 14)
    ax.set_zlabel(name, fontsize = 14)
    ax.view_init(elev = 18, azim = 51)
    
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    ax.zaxis.set_tick_params(labelsize=8)
    
    ax.tick_params(axis='both', which='major', pad=.5)
    
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.zaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    
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
    sns.heatmap(A*abs(Theta[k,:,:]), mask = mask, ax = ax, square = True, cmap = this_cmap, vmin = -.1, vmax = .1, linewidths=.5, cbar = False)
    ax.set_title(f"Precision matrix at timestamp {k}")
    
    return 

def single_heatmap_animation(Theta, method = 'truth', save = False):
    
    (K,p,p) = Theta.shape
    
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    fargs = (Theta, method, ax,)
    
    def _init():
        #ax.cla()
        A = np.zeros((p, p))
        mask = (A == 0) 
        sns.heatmap(A, mask = mask, ax = ax, square = True, cmap = 'Blues', vmin = 0, vmax = 1, linewidths=.5, cbar = False)

    anim = FuncAnimation(fig, plot_single_heatmap, frames = K, init_func= _init, interval= 1000, fargs = fargs, repeat = True)
    
    if save:    
        anim.save("single_network.gif", writer='imagemagick')
        
    return anim


def plot_multiple_heatmap(k, Theta, results, axs):
    
    plot_single_heatmap(k, Theta, 'truth', axs[0,0])
    plot_single_heatmap(k, results.get('ADMM').get('Theta'), 'ADMM', axs[0,1])
    plot_single_heatmap(k, results.get('LTGL').get('Theta'), 'LTGL', axs[1,0])
    plot_single_heatmap(k, results.get('SGL').get('Theta'), 'SGL', axs[1,1])
    
    return

def multiple_heatmap_animation(Theta, results, save = False):
    (K,p,p) = Theta.shape
    fig, axs = plt.subplots(nrows = 2, ncols=2, figsize=(10,10))

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

