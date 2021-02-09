'''
Author: https://github.com/rahuln/lvsglasso

'''

from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from numpy.linalg import norm

from gglasso.Lvsglasso.lvsglasso.proximal import block_thresh, block_thresh_2d, block_thresh_entrywise, prox_nuc
from gglasso.Lvsglasso.lvsglasso.util import pack_and_stack, partial_coher

import operator
from itertools import product
import collections


def lvsglasso_obj(SX_, R_, S_, ewR_, ewL_, lamS, lamL):
    F = S_.shape[0]
    S_gl = np.sum(1. / np.sqrt(F) * np.linalg.norm(S_, axis=0))
    trace_term = 0.
    for f in range(F):
        trace_term += np.sum(SX_[f].T * R_[f]).real
    return (trace_term - np.sum(np.log(ewR_))
            + lamS * S_gl
            + lamL * np.sum(ewL_))


def admm_consensus(SX_, lamS, lamL, niter=100, alpha=1., mu=None,
                   mu_cont=None, mu_cont_iter=10, mu_min=1e-6,
                   S_init=None, L_init=None,
                   abstol=1e-5, reltol=1e-5, verbose=False, compute_obj=False,
                   compute_infeas=False, do_lowrank=True, do_sparse=True):
    """ ADMM algorithm for sparse plus low-rank Gaussian graphical model.
        Algorithm based off of Section 3 of "Alternating direction methods for
        latent variable graphical model selection".

        Note: R[] := S[] - L[]

        SX_ : Estimate of spectral density matrix.
        lamS : Regularization parameter for sparse component.
        lamL : Regularization parameter for low-rank component.
        S_init : Initialization for S_.
        L_init : Initiialization for L_.
        niter : Maximum number of iterations.
        alpha : Overrelaxation parameter \in (0,2], but values in [1.5, 1.8]
                are suggested.
        mu : ADMM inverse dual step-size.
        mu_cont : Continuation parameter for mu (how much to reduce by).
                  If None then no continuation. Reasonable values are 0.5 or
                  0.25.
        mu_cont_iter : Number of iterations between performing mu-continutation.
        mu_min : Smallest value allowed for mu.
        abstol : Absolute stopping tolerance.
        reltol : Relative stopping tolerance.
        verbose : Whether to print progress.
        compute_obj : Whether to compute objective function at each iteration.
        compute_infeas : Whether or not to compute relative infeasability at each iteration.
        do_lowrank : Whether or not to perform the low-rank updates (can be set to False to run
            like sparse model).
        do_sparse : Whether or not to perform the sparse updates.

        Returns:
        
        S : Estimated sparse component.
        L : Estimated low-rank component.
        obj : Values of objective function.
        infeas : Values of infeasibility.
        diagnostics : Dictionary with diagnostic information including:
                        r_norm, eps_pri, s_norm, eps_dual, and `converged` flag
                        indicating if algorithm converged.
        lamS : Regularization parameter that was used for the sparse component.
        lamL : Regularization parameter that was used for the low-rank component.

        Notes:
          - mu is 1/\rho where \rho is the usual dual step-size parameter.

    """
    if mu_cont is not None and mu_cont < 0. and mu_cont > 1.:
        raise RuntimeError("mu_cont must be in (0, 1]")

    hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.2g"

    F, p, _ = SX_.shape

    if mu is None:
        mu = p

    R_ = np.empty((F, p, p), dtype=np.complex128)
    RY_ = np.empty_like(R_)
    S_ = np.empty((F, p, p), dtype=np.complex128)
    SY_ = np.empty_like(S_)

    if S_init is not None:
        S_ = S_init.copy()
        SY_ = S_.copy()
    else:
        for f in range(F):
            # Avoid divide-by-zero in block_thresh on first iteration
            S_[f] = np.eye(p)
            SY_[f] = np.eye(p)

    if L_init is not None:
        L_ = L_init.copy()
        LY_ = L_.copy()
    else:
        L_ = np.zeros((F, p, p), dtype=np.complex128)
        LY_ = np.zeros_like(L_)

    if S_init is None or L_init is None:
        for f in range(F):
            R_[f] = np.eye(p)
            RY_[f] = np.eye(p)
    else:
        R_ = S_ - L_
        RY_ = R_.copy()

    # Initialize here in case we're not updating it
    ewL_ = np.zeros((F, p))

    Y = pack_and_stack(RY_, SY_, LY_)
    X = np.empty_like(Y)
    U = np.empty_like(Y)

    U_R = np.zeros_like(R_)
    U_S = np.zeros_like(S_)
    U_L = np.zeros_like(L_)

    obj = np.zeros(niter)
    infeas = np.zeros(niter)
    r_norm = np.zeros(niter)
    s_norm = np.zeros(niter)
    eps_pri = np.zeros(niter)
    eps_dual = np.zeros(niter)

    converged = False

    if verbose:
        print(hdr_fmt %
              ("it", "obj", "infeas", "r_norm", "eps_pri", "s_norm", "eps_dual", "mu"))

    # Do one "iter" first so that B_S and B_L are updated before doing their
    # proximal updates.
    B_R = RY_ + mu * U_R
    B_S = SY_ + mu * U_S
    B_L = LY_ + mu * U_L

    # Update R
    tmp = mu * SX_ - B_R
    ew_, ev_ = np.linalg.eigh(tmp)
    ewR_ = (-ew_ + np.sqrt(ew_ ** 2 + 4 * mu)) / 2
    for f in range(F):
        ev_f = ev_[f]
        R_[f] = np.dot(ev_f, np.dot(np.diag(ewR_[f]), ev_f.conj().T))

    # Relaxation
    RO_ = alpha * R_ + (1 - alpha) * RY_
    SO_ = alpha * S_ + (1 - alpha) * SY_
    LO_ = alpha * L_ + (1 - alpha) * LY_

    B_R = RO_ - mu * U_R
    B_S = SO_ - mu * U_S
    B_L = LO_ - mu * U_L

    # Update tilde variables
    Gam = -(B_R - B_S + B_L) / 3.
    RY_ = B_R + Gam
    if do_sparse:
        SY_ = B_S - Gam
    if do_lowrank:
        LY_ = B_L + Gam

    # Update dual variable (Lagrange multiplier)
    U_R = U_R - (RO_ - RY_) / mu
    if do_sparse:
        U_S = U_S - (SO_ - SY_) / mu
    if do_lowrank:
        U_L = U_L - (LO_ - LY_) / mu

    # Now start actual iterations
    for it in range(1, niter):

        B_R = RY_ + mu * U_R
        B_S = SY_ + mu * U_S
        B_L = LY_ + mu * U_L

        # Update R
        tmp = mu * SX_ - B_R
        ew_, ev_ = np.linalg.eigh(tmp)
        ewR_ = (-ew_ + np.sqrt(ew_ ** 2 + 4 * mu)) / 2
        for f in range(F):
            ev_f = ev_[f]
            R_[f] = np.dot(ev_f, np.dot(np.diag(ewR_[f]), ev_f.conj().T))

        # Update S
        if do_sparse:
            S_ = block_thresh(B_S, lamS * mu)

        # Update L
        if do_lowrank:
            # Original singular-value thresholding on each frequency
            ew_, ev_ = np.linalg.eigh(B_L)
            ewL_ = prox_nuc(ew_, lamL * mu)
            for f in range(F):
                ev_f = ev_[f]
                L_[f] = np.dot(ev_f, np.dot(np.diag(ewL_[f]), ev_f.conj().T))

        pack_and_stack(R_, S_, L_, out=X)

        # Relaxation
        RO_ = alpha * R_ + (1 - alpha) * RY_
        SO_ = alpha * S_ + (1 - alpha) * SY_
        LO_ = alpha * L_ + (1 - alpha) * LY_

        Y_old = Y.copy()
        B_R = RO_ - mu * U_R
        B_S = SO_ - mu * U_S
        B_L = LO_ - mu * U_L

        # Update tilde variables
        Gam = -(B_R - B_S + B_L) / 3.
        RY_ = B_R + Gam
        if do_sparse:
            SY_ = B_S - Gam
        if do_lowrank:
            LY_ = B_L + Gam

        pack_and_stack(RY_, SY_, LY_, out=Y)

        # Update dual variable (Lagrange multiplier)
        U_R = U_R - (RO_ - RY_) / mu
        if do_sparse:
            U_S = U_S - (SO_ - SY_) / mu
        if do_lowrank:
            U_L = U_L - (LO_ - LY_) / mu

        pack_and_stack(U_R, U_S, U_L, out=U)

        # Objective function
        if compute_obj:
            obj[it] = lvsglasso_obj(SX_, R_, S_, ewR_, ewL_, lamS, lamL)

        # Infeasibility metric
        if True or compute_infeas:
            infeas[it] = 0.
            for f in range(F):
                infeas[it] += (norm(R_[f] - S_[f] + L_[f], 'fro') /
                               np.max([1., norm(R_[f], 'fro'), norm(S_[f], 'fro'),
                                       norm(L_[f], 'fro')]))

        r_norm[it] = 0.
        s_norm[it] = 0.
        for f in range(F):
            r_norm[it] += norm((R_[f] - S_[f] + L_[f]), 'fro')
            s_norm[it] += norm(-(Y[f] - Y_old[f]) / mu, 'fro')

        eps_pri[it] = np.sqrt(p * p * F) * abstol
        eps_dual[it] = np.sqrt(p * p * F) * abstol
        for f in range(F):
            eps_pri[it] += reltol * np.maximum(norm(X[f], 'fro'),
                                               norm(Y[f], 'fro'))
            eps_dual[it] += reltol * norm(U[f], 'fro')

        if it % mu_cont_iter == 0 and mu_cont is not None:
            mu = np.maximum(mu * mu_cont, mu_min)

        if verbose and it % 10 == 0:
            print(out_fmt % (it, obj[it], infeas[it], r_norm[it], eps_pri[it],
                             s_norm[it], eps_dual[it], mu))

        if r_norm[it] < eps_pri[it] and s_norm[it] < eps_dual[it]:
            converged = True
            break

    diagnostics = {'r_norm': r_norm, 'eps_pri': eps_pri, 's_norm': s_norm,
                   'eps_dual': eps_dual, 'converged': converged}

    inds = np.s_[1:(it + 1)]
    return S_, L_, obj[inds], infeas[inds], diagnostics, lamS, lamL


def admm_prox(SX_, lamS, lamL, mu=1., niter=100, tau=0.5,
              mu_cont=None, mu_cont_iter=10, mu_min=1e-6,
              S_init=None, L_init=None,
              abstol=1e-5, reltol=1e-5, stoptol=1e-5, verbose=False,
              compute_obj=False, compute_infeas=False, do_lowrank=True,
              do_sparse=True):
    """ ADMM algorithm for sparse plus low-rank graphical model for time series
        (sparse inverse-spectral density matrices). The algorithm is based off
        of Section 4 of "Alternating direction methods for latent variable
        graphical model selection" and uses a proximal-gradient method to solve
        an intractable sub-problem.

        Note: R[] := S[] - L[]

        SX_ : Estimate of spectral density matrix.
        lamS : Regularization parameter for sparse component.
        lamL : Regularization parameter for low-rank component.
        mu : ADMM inverse dual step-size.
        mu_cont : Continuation parameter for mu (how much to reduce by).
                  If None then no continuation. Reasonable values are 0.5 or
                  0.25.
        mu_cont_iter : Number of iterations between performing mu-continutation.
        mu_min : Smallest value allowed for mu.
        abstol : Absolute stopping tolerance.
        reltol : Relative stopping tolerance.
        stoptol : Tolerance for checking objective. This may be deprecated...
        verbose : Whether to print progress.
        compute_obj : Whether to compute objective function at each iteration.

        Returns:
        
        S : Estimated sparse component.
        L : Estimated low-rank component.
        obj : Values of objective function.
        infeas : Values of infeasibility.
        diagnostics : Dictionary with diagnostic information including:
                        r_norm, eps_pri, s_norm, eps_dual, and `converged` flag
                        indicating if algorithm converged.

        Notes:
          - mu is 1/\rho where \rho is the usual dual step-size in the
            literature.
    """

    print("PROXIMAL VERSION")

    if mu_cont is not None and mu_cont < 0. and mu_cont > 1.:
        raise RuntimeError("mu_cont must be in (0, 1]")

    hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g"

    F, p, _ = SX_.shape

    R_ = np.empty((F, p, p), dtype=np.complex128)
    S_ = np.empty((F, p, p), dtype=np.complex128)

    if S_init is not None:
        S_ = S_init.copy()
        SY_ = S_.copy()
    else:
        for f in range(F):
            # Avoid divide-by-zero in block_thresh on first iteration
            S_[f] = np.eye(p)

    if L_init is not None:
        L_ = L_init.copy()
    else:
        L_ = np.zeros((F, p, p), dtype=np.complex128)

    if S_init is None or L_init is None:
        for f in range(F):
            R_[f] = np.eye(p)
    else:
        R_ = S_ - L_

    # Initialize here in case we're not updating it
    ewL_ = np.zeros((F, p))

    U_ = np.zeros((F, p, p), dtype=np.complex128)

    obj = np.zeros(niter)
    infeas = np.zeros(niter)
    r_norm = np.zeros(niter)
    s_norm = np.zeros(niter)
    eps_pri = np.zeros(niter)
    eps_dual = np.zeros(niter)

    if verbose:
        print(hdr_fmt %
              ("it", "obj", "infeas", "r_norm", "eps_pri", "s_norm", "eps_dual"))

    converged = False
    for it in range(1, niter):
        # Update R_
        B = mu * SX_ - mu * U_ - S_ + L_
        ew_, ev_ = np.linalg.eigh(B)
        ewR_ = (-ew_ + np.sqrt(ew_ ** 2 + 4 * mu)) / 2
        for f in range(F):
            ev_f = ev_[f]
            R_[f] = np.dot(ev_f, np.dot(np.diag(ewR_[f]), ev_f.conj().T))

        S_old = S_.copy()
        L_old = L_.copy()

        # Update S_ and L_ via a step of proximal gradient-descent
        GR = S_ - L_ - R_ + mu * U_

        if do_sparse:
            G_ = S_ - tau * GR
            S_ = block_thresh(G_, tau * mu * lamS)

        if do_lowrank:
            H_ = L_ + tau * GR
            ew_, ev_ = np.linalg.eigh(H_)
            ewL_ = prox_nuc(ew_, tau * mu * lamL)
            for f in range(F):
                ev_f = ev_[f]
                L_[f] = np.dot(ev_f, np.dot(np.diag(ewL_[f]), ev_f.conj().T))

        # Update U_
        U_ = U_ - (R_ - S_ + L_) / mu

        # Objective function
        if compute_obj:
            obj[it] = lvsglasso_obj(SX_, R_, S_, ewR_, ewL_, lamS, lamL)

        # Infeasibility metric
        if compute_infeas:
            infeas[it] = 0.
            for f in range(F):
                infeas[it] += (norm(R_[f] - S_[f] + L_[f], 'fro') /
                               np.max([1., norm(R_[f], 'fro'), norm(S_[f], 'fro'),
                                       norm(L_[f], 'fro')]))

        r_norm[it] = 0.
        s_norm[it] = 0.
        for f in range(F):
            r_norm[it] += norm((R_[f] - S_[f] + L_[f]), 'fro')
            s_norm[it] += norm(-(np.vstack((S_[f], L_[f])) -
                                 np.vstack((S_old[f], L_old[f]))) / mu, 'fro')

        # There's no F in the epsilons b/c the objective has been scaled to
        # remove the 1/F factors, effectively scaling the norms already.
        eps_pri[it] = np.sqrt(F * p * p) * abstol
        eps_dual[it] = np.sqrt(F * p * p) * abstol
        for f in range(F):
            eps_pri[it] += reltol * np.maximum(norm(R_[f], 'fro'),
                                               norm(np.vstack((S_[f], -L_[f])), 'fro'))
            eps_dual[it] += reltol * norm(U_[f], 'fro')

        if it % mu_cont_iter == 0 and mu_cont is not None:
            mu = np.maximum(mu * mu_cont, mu_min)

        if verbose and it % 10 == 0:
            print(out_fmt % (it, obj[it], infeas[it], r_norm[it], eps_pri[it],
                             s_norm[it], eps_dual[it]))

        if r_norm[it] < eps_pri[it] and s_norm[it] < eps_dual[it]:
            converged = True
            break

    diagnostics = {'r_norm': r_norm, 'eps_pri': eps_pri, 's_norm': s_norm,
                   'eps_dual': eps_dual, 'converged': converged}

    inds = np.s_[1:(it + 1)]
    return S_, L_, obj[inds], infeas[inds], diagnostics, lamS, lamL


def AIC(SX_, S_, L_, ei_thresh=1e-6):
    """
    AIC = trace(SX*R) - sum(log(ewR)) + 2*F*n_edges + rank
    """
    F, p, _ = S_.shape
    R_ = S_ - L_
    rank = sum([np.linalg.matrix_rank(L_[f]) for f in range(L_.shape[0])])
    n_edges = np.sum(np.max(partial_coher(S_), axis=0) > ei_thresh) - S_.shape[1]
    ewR_ = np.linalg.eigvalsh(R_)
    aic = 0.
    for f in range(F):
        aic += 2 * np.sum(SX_[f].T * R_[f]).real
    aic = aic - 2 * np.sum(np.log(ewR_) + (F * (n_edges / 2) + p * rank))
    return aic


def BIC(SX_, S_, L_, n_samples, ei_thresh=1e-6):
    return np.sum(BIC_split(SX_, S_, L_, n_samples, ei_thresh=ei_thresh))


def BIC_split(SX_, S_, L_, n_samples, ei_thresh=1e-6):
    """
    BIC = trace(SX*R) - sum(log(ewR)) + (2*F*n_edges + rank) * n_samples
    """
    F, p, _ = S_.shape
    R_ = S_ - L_

    ranks = [np.linalg.matrix_rank(L_[f]) for f in range(L_.shape[0])]
    rank = np.sum(ranks)

    n_edges = np.sum(np.max(partial_coher(S_), axis=0) > ei_thresh) - S_.shape[1]

    ewR_ = np.linalg.eigvalsh(R_)

    likelihood = 0.
    for f in range(F):
        likelihood += 2 * np.sum(SX_[f].T * R_[f]).real

    likelihood = likelihood - 2 * np.sum(np.log(ewR_))
    penalty = ((n_edges / 2) + (p / F) * rank) * np.log(n_samples)

    return likelihood, penalty
