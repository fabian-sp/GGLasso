"""
Functional Graphical Lasso experiment
===========================================

In this example, we want to explain how Functional Graphical Lasso [ref13]_ works and how you can make use of it with ``GGLasso``. 
For this tutorial, we use the ``scikit-fda`` package.

For Functional Graphical Lasso **every variable is representing a function or time series.** 
In order to obtain a finite-dimensional problem, we represent the function in some basis (e.g. by computing Fourier coefficients and truncating). 
Then, we compute correlations of the corresponding coefficients and are interested in the relationships between different functional variables.
 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from skfda.datasets import make_gaussian_process
from skfda.misc.covariances import *
from skfda.misc.metrics import l2_distance, l2_norm
from skfda.representation.basis import BSpline, Fourier
from skfda.representation.grid import FDataGrid


#%%
# We work in the following setup: we create (functional) variables, called :math:`v_0, v_1, \dots`. Some of those will come from a Gaussian process, some will be constructed in particular ways. 
# We then draw samples for each variable. 
# In the next step, we will compute representations with respect to some basis (e.g. Fourier or BSplines). 
# Here, we choose the Fourier basis.

t0 = 0
t1 = 4

n_ts = 101
n_samples = 10**2

_t = np.linspace(t0,t1,n_ts)

def get_basis(style, n_components):
    if style == 'bspline':
        basis = BSpline(domain_range=(t0, t1), n_basis=n_components, order=4)
    elif style == 'fourier':
        basis = Fourier(domain_range=(t0, t1), n_basis=n_components, period=None)
    return basis

style = 'fourier'

#%%
# Generate samples from a simple function.

def hat(t,A): 
    y = (t<= t1/2) * (2*A/t1 * t) + (t> t1/2) *(2*A - 2*A/t1 * t)
    return y

modes = 4*np.random.rand(n_samples) - 2
v2 = np.zeros((n_samples, n_ts))

for j in range(n_samples):
    v2[j,:] = hat(_t, modes[j])


#%%
# Define variables from Gaussian process.

cov0 = Gaussian(variance=1., length_scale=1.)
cov1 = Gaussian(variance=1, length_scale=0.5)

all_var = [cov0, cov1, v2]
n_var = len(all_var)


#%%
# Sample time series. The first two come from a Gaussian process where the respective kernels have different length scales. 
# The third one is simply a linearly increasing and the decreasing function.


samples = list()
labels = list()

for j in np.arange(n_var):
    if not isinstance(all_var[j], np.ndarray):
        _ds = make_gaussian_process(
                n_samples=n_samples,
                n_features=n_ts,
                start=t0,
                stop=t1,
                cov=all_var[j], mean=0, random_state=20)
    else:
        _ds = FDataGrid(all_var[j], _t)
    
    _lb = np.array([f'v{j}'] * n_samples)
    
    samples.append(_ds)
    labels.append(_lb)

#%%
# We want to create a fourth variable that is constructed using the first variable :math:`v_0` as follows: 
# 
# * permute the coefficients of the basis representation
# * retransform to a time series, but with the basis elements not permuted. 
#
# This creates a variable where the relationship to the original :math:`v_0` is not obvious if only looking at the time series.

n_comp = 9

basis = get_basis(style, n_comp)
_traf = samples[0].to_basis(basis)   

rng = np.random.default_rng(1917)
_perm = rng.permutation(n_comp)
_traf.coefficients = _traf.coefficients[:, _perm]

v3_ds = _traf.to_grid(samples[0].grid_points)

n_var += 1

samples.append(v3_ds)
_lb3 = np.array(['v3'] * n_samples)
labels.append(_lb3)

#%%
# We can now plot the sampled time series for each variable. 

colors = ['darkred', 'C1', 'grey', 'steelblue']
_alpha = 0.2
_lw = 2.

fig, axs = plt.subplots(n_var,1, figsize=(10,12))

for j in range(n_var):
    ax = axs.ravel()[j]
    samples[j].plot(axes=ax, group=labels[j], group_colors={f'v{j}': colors[j]}, alpha=_alpha, lw=_lw)
    ax.set_title(f'Sampled time series for v{j}')

fig.tight_layout()

#%%
# Next steps:
# 
# * compute a basis representation (for a finite number of basis components!). This is done using the scikit-fda function ``.to_basis()``.
# * reconstruct the time series. This is done using the scikit-fda function ``.to_grid()``.
# * compute the reconstruction error (defined as the median relative :math:`ell_2` error).
# * plot the reconstructed time series and the error. 

q = 7
fig, axs = plt.subplots(n_var, 2, figsize=(15,9))

for j in range(n_var):
    basis = get_basis(style, q)
    traf = samples[j].to_basis(basis)
        
    recov_ds = traf.to_grid(samples[j].grid_points)
    diff = samples[j] - recov_ds
        
    ax = axs[j,0]
    recov_ds.plot(axes=ax, group=labels[j], group_colors={f'v{j}': colors[j]}, alpha=_alpha, lw=_lw)
    ax.set_title(f'Basis representation - reconstructed time series for v{j}')
    
    ax2 = axs[j,1]
    diff.plot(axes=ax2, group=labels[j], group_colors={f'v{j}': colors[j]}, alpha=_alpha, lw=_lw)
    ax2.set_title(f'Reconstruction error for v{j}')
    ax2.set_ylim(ax.get_ylim())
    
fig.tight_layout()

#%%
# Now, we compute the reconstruction error for an increasing number of basis components. 
# We would expect the error to go to zero - if we have chosen a suitable basis (e.g. Fourier for periodic functions).


fig, ax = plt.subplots()
    
for j in range(n_var):
    all_err = list()
    
    if style != 'fourier':
        all_q = range(4,12)
    else:
        all_q = [5,7,9,11,13] # Fourier has always odd number of basis elements
        
    for k in range(len(all_q)):
        basis = get_basis(style, all_q[k])
        this_traf = samples[j].to_basis(basis)
        
        recov_ds = this_traf.to_grid(samples[j].grid_points)
    
        this_err = l2_distance(samples[j], recov_ds) / l2_norm(samples[j])
        all_err.append(np.median(this_err))

   
    ax.plot(all_q, all_err, c=colors[j], lw = 4, marker='p', markersize=10, markeredgecolor='k', alpha=0.8, label=f'v{j}')
    
ax.set_xlabel('Number of basis components')
ax.set_ylabel('Median l2 error')
ax.grid(ls = '-', lw = .5) 
ax.legend()

#%%
# Functional Graphical Lasso (FSGL) computation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# As seen above, with sufficiently many basis components we have a good representation of the original function.
# Now, compute the coefficients for the first :math:`n_{comp}` basis components. Then, compute their correlations which will be the input for Functional Graphical Lasso.
# 
# If we choose :math:`n_{comp}=9`, we expect :math:`4\cdot9=36` variables in total.

from gglasso.helper.basic_linalg import scale_array_by_diagonal
from gglasso.helper.utils import lambda_max_fsgl, frob_norm_per_block
from gglasso.solver.functional_sgl_admm import ADMM_FSGL
from gglasso.helper.experiment_helper import plot_fsgl_heatmap

n_comp = 9
p = n_var
M = n_comp
pM = p*M
all_traf = list()
    
for j in range(n_var):
    basis = get_basis(style, n_comp)
    _traf = samples[j].to_basis(basis)
    
    all_traf.append(_traf.coefficients)
    
Z = np.hstack(all_traf)
print("(N,p) = ", Z.shape)

# compute correlations
S = np.cov(Z.T)
S = scale_array_by_diagonal(S)

print("S has shape ", S.shape)


#%%
# Plot the input data for FSGL.

# plot samples
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(Z, ax=ax, alpha=1, vmin=-3, vmax=3, cmap='coolwarm', cbar=True, linewidth = 0.005, linecolor = 'lightgrey',)
ax.vlines([(j+1)*n_comp for j in range(n_var)], 0, n_samples, color='k', lw=4)
ax.set_title('Basis coefficients for all samples and variables')

# plot heatmap
fig, axs = plt.subplots(1,2,figsize=(9,4), gridspec_kw={'width_ratios': [5.8,4.5]})
_v = 1.
ax = axs[0]
plot_fsgl_heatmap(S, p, M, ax=ax)
ax.set_title('Empirical correlation matrix')

# plot Frobenius norm of each subblock
ax = axs[1]
to_plot = np.round(frob_norm_per_block(S, n_comp),3)
sns.heatmap(to_plot, annot=True, vmin=0, vmax=to_plot.max(), linewidth=0.005, linecolor='w', cbar=False, ax=ax)
ax.set_title(r"Heatmap of $\||\Theta^M_{jl}\||_F$")

#%%
# For FSGL, we have to specify a regularization parameter :math:`\lambda`. 
# Typically, one computes the solution for a range of :math:`\lambda` values and then chooses the best suited solution according to some criterion.

lambda_max = lambda_max_fsgl(S, M)
lambda_min = 0.1 * lambda_max
lambda_range = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 9)[::-1]

# solve Functional Graphical Lasso for all lambda
Omega_0 = np.eye(pM)
all_sol = dict()

for j in range(len(lambda_range)):
    _lam = lambda_range[j]
    sol, info = ADMM_FSGL(S, _lam, M, Omega_0, rho=1., max_iter=2000, tol=1e-8, rtol=1e-7, verbose=False, measure=True)
    
    Omega_0 = sol['Omega'].copy() # warm start
    all_sol[_lam] = sol.copy()

#%%
# As :math:`v_3` was constructed from :math:`v_0`, we would expect that their relationship can be recovered. Also, :math:`v_0`and :math:`v_1` are more related than :math:`v_0` and :math:`v_2` as both come from a Gaussian process whereas :math:`v_2` was piecewise linear.
# Let's see whether these relationships are correctly identified by FSGL. 
# Two variables :math:`v_j` and :math:`v_l` are associated if and only if the corresponding block :math:`\Theta^M_{jl}` **is non-zero**. If the basis represenation is exact, this result is given in Lemma 1 in [ref13]_ .
# The block itself could be sparse or dense and its individual entries are harder to interpret.

fig, axs = plt.subplots(3,3, figsize=(17,15))

for j in range(len(lambda_range)):
    _lam = lambda_range[j]
    ax = axs.ravel()[j]
    plot_fsgl_heatmap(all_sol[_lam]['Theta'], p, M, ax=ax)
    ax.set_title(fr"$\lambda$={_lam}")
    
fig.suptitle(r'FSGL solution for all $\lambda$-values')

#%% 
# Finally, we plot the Frobenius norm of each subblock for each :math:`\lambda`-value. 
# Like this, we can see which block first (i.e. for the largest :math:`\lambda`) enters the solution. As expected it is the block encoding the relationship between :math:`v_0` and :math:`v_3`.

Fnorm = np.zeros((len(lambda_range), p, p))
for j in range(len(lambda_range)):
    _lam = lambda_range[j]
    Fnorm[j,:,:] = frob_norm_per_block(all_sol[_lam]['Theta'], M)

Fnorm[Fnorm <= 1e-8] = np.nan

fig, ax = plt.subplots()
for i in range(p):
    for j in np.arange(start=i+1, stop=p):
        ax.plot(lambda_range, Fnorm[:,i,j], lw=3, marker='o', markersize=8, alpha=0.7, markevery=(1,i+1),
                markeredgecolor='k',
                label = f'block {i}_{j}')

ax.set_xscale('log')
ax.set_ylim(0,)
ax.set_xlim(lambda_range[-1], lambda_range[0])

ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r"$\||\Theta^M_{jl}\||_F$", fontsize=14)
ax.grid(ls = '-', lw = .5) 
ax.legend()



