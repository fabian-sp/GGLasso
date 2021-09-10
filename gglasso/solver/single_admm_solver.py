"""
author: Fabian Schaipp
"""

import numpy as np
import time
from scipy.sparse.csgraph import connected_components
from scipy.linalg import block_diag
import warnings

from .ggl_helper import prox_od_1norm, phiplus, prox_rank_norm


def ADMM_SGL(S, lambda1, Omega_0, Theta_0=np.array([]), X_0=np.array([]),
             rho=1., max_iter=1000, tol=1e-7, rtol=1e-4, stopping_criterion='boyd',\
             update_rho=True, verbose=False, measure=False, latent=False, mu1=None):
    """
    This is an ADMM solver for the (Latent variable) Single Graphical Lasso problem (SGL).
    If ``latent=False``, this function solves

    .. math::
       \min_{\Omega, \Theta \in \mathbb{S}^p_{++}} - \log \det \Omega + \mathrm{Tr}(S\Omega) + \lambda \|\Theta\|_{1,od}

       s.t. \quad \Omega = \Theta

    If ``latent=True``, this function solves

    .. math::
       \min_{\Omega, \Theta, L \in \mathbb{S}^p_{++}} - \log \det (\Omega) + \mathrm{Tr}(S \Omega) + \lambda_1 \|\Theta\|_{1,od} + \mu_1 \|L\|_{\star}

       s.t. \quad \Omega = \Theta - L

    Note:
        * Typically, ``sol['Omega']`` is positive definite and ``sol['Theta']`` is sparse.
        * We use scaled ADMM, i.e. X are the scaled (with 1/rho) dual variables for the equality constraint.
    Parameters
    ----------
    S : array (p,p)
        empirical covariance matrix. Needs to be symmetric and positive semidefinite.
    lambda1 : float, positive
        sparsity regularization parameter.
    Omega_0 : array (p,p)
        starting point for the Omega variable. Choose ``np.eye(p)`` if no better starting point is known.
    Theta_0 : array (p,p), optional
        starting point for the Theta variable. If not specified, it is set to the same as Omega_0.
    X_0 : array (p,p), optional
        starting point for the X variable. If not specified, it is set to zero array.
    rho : float, positive, optional
        step size paramater for the augmented Lagrangian in ADMM. The default is 1. Tune this parameter for optimal performance.
    max_iter : int, optional
        maximum number of iterations. The default is 1000.
    tol : float, positive, optional
        tolerance for the primal residual. See "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", Boyd et al. for details.
        The default is 1e-7.
    rtol : float, positive, optional
        tolerance for the dual residual. The default is 1e-4.
    stopping_criterion : str, optional

        * 'boyd': Stopping criterion after Boyd et al.
        * 'kkt': KKT residual is chosen as stopping criterion. This is computationally expensive to compute.

        The default is 'boyd'.
    update_rho : boolean, optional
        Whether the penalty parameter ``rho`` is updated, see Boyd et al. page 20-21 for details. The default is True.
    verbose : boolean, optional
        verbosity of the solver. The default is False.
    measure : boolean, optional
        turn on/off measurements of runtime per iteration. The default is False.
    latent : boolean, optional
        Solve the SGL with or without latent variables (see above for the exact formulations).
        The default is False.
    mu1 : float, positive, optional
        low-rank regularization parameter. Only needs to be specified if latent=True.
    Returns
    -------
    sol : dict
        contains the solution, i.e. Omega, Theta, X (and L if ``latent=True``) after termination. All elements are (p,p) arrays.
    info : dict
        status and measurement information from the solver.
    """
    assert Omega_0.shape == S.shape
    assert S.shape[0] == S.shape[1]
    assert lambda1 > 0

    assert stopping_criterion in ["boyd", "kkt"]

    if latent:
        assert mu1 is not None
        assert mu1 > 0

    (p, p) = S.shape

    assert rho > 0, "ADMM penalization parameter must be positive."

    # initialize
    Omega_t = Omega_0.copy()

    if len(Theta_0) == 0:
        Theta_0 = Omega_0.copy()
    if len(X_0) == 0:
        X_0 = np.zeros((p, p))

    Theta_t = Theta_0.copy()
    L_t = np.zeros((p, p))
    X_t = X_0.copy()

    runtime = np.zeros(max_iter)
    residual = np.zeros(max_iter)
    status = ''


    if verbose:
        print("------------ADMM Algorithm for Single Graphical Lasso----------------")

        if stopping_criterion == 'boyd':
            hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s"
            out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
            print(hdr_fmt % ("iter", "r_t", "s_t", "eps_pri", "eps_dual"))
        elif stopping_criterion == 'kkt':
            hdr_fmt = "%4s\t%10s"
            out_fmt = "%4d\t%10.4g"
            print(hdr_fmt % ("iter", "kkt residual"))

    ##################################################################
    ### MAIN LOOP STARTS
    ##################################################################
    for iter_t in np.arange(max_iter):
        if measure:
            start = time.time()


        # Omega Update
        W_t = Theta_t - L_t - X_t - (1 / rho) * S
        eigD, eigQ = np.linalg.eigh(W_t)
        Omega_t_1 = Omega_t.copy()
        Omega_t = phiplus(beta=1 / rho, D=eigD, Q=eigQ)

        # Theta Update
        Theta_t = prox_od_1norm(Omega_t + L_t + X_t, (1 / rho) * lambda1)

        # L Update
        if latent:
            C_t = Theta_t - X_t - Omega_t
            # C_t = (C_t.T + C_t)/2
            eigD1, eigQ1 = np.linalg.eigh(C_t)
            L_t = prox_rank_norm(C_t, mu1/rho, D=eigD1, Q=eigQ1)

        # X Update
        X_t = X_t + Omega_t - Theta_t + L_t

        
        
        if measure:
            end = time.time()
            runtime[iter_t] = end - start

        # Stopping criterion
        if stopping_criterion == 'boyd':
            r_t,s_t,e_pri,e_dual = ADMM_stopping_criterion(Omega_t, Omega_t_1, Theta_t, L_t, X_t,\
                                                           S, rho, tol, rtol, latent)
            
            # update rho
            if update_rho:
                if r_t >= 10*s_t:
                    rho_new = 2*rho
                elif s_t >= 10*r_t:
                    rho_new = 0.5*rho
                else:
                    rho_new = 1.*rho
                
                # rescale dual variables
                X_t = (rho/rho_new)*X_t
                rho = rho_new
                
                
            residual[iter_t] = max(r_t,s_t)

            if verbose:
                print(out_fmt % (iter_t,r_t,s_t,e_pri,e_dual))
            if (r_t <= e_pri) and  (s_t <= e_dual):
                status = 'optimal'
                break

        elif stopping_criterion == 'kkt':
            eta_A = kkt_stopping_criterion(Omega_t, Theta_t, L_t, rho * X_t, S, lambda1, latent, mu1)
            residual[iter_t] = eta_A

            if verbose:
                print(out_fmt % (iter_t,eta_A))
            if eta_A <= tol:
                status = 'optimal'
                break


    ##################################################################
    ### MAIN LOOP FINISHED
    ##################################################################

    # retrieve status (partially optimal or max iter)
    if status != 'optimal':
        if stopping_criterion == 'boyd':
            if (r_t <= e_pri):
                status = 'primal optimal'
            elif (s_t <= e_dual):
                status = 'dual optimal'
            else:
                status = 'max iterations reached'
        else:
            status = 'max iterations reached'

    print(f"ADMM terminated after {iter_t+1} iterations with status: {status}.")

    ### CHECK FOR SYMMETRY
    if abs((Omega_t).T - Omega_t).max() > 1e-5:
        warnings.warn(f"Omega variable is not symmetric, largest deviation is {abs((Omega_t).T - Omega_t).max()}.")
    
    if abs((Theta_t).T - Theta_t).max() > 1e-5:
        warnings.warn(f"Theta variable is not symmetric, largest deviation is {abs((Theta_t).T - Theta_t).max()}.")
    
    if abs((L_t).T - L_t).max() > 1e-5:
        warnings.warn(f"L variable is not symmetric, largest deviation is {abs((L_t).T - L_t).max()}.")

    ### CHECK FOR POSDEF
    D = np.linalg.eigvalsh(Theta_t - L_t)
    if D.min() <= 0:
        print(
            f"WARNING: Theta (Theta - L resp.) is not positive definite. Solve to higher accuracy! (min EV is {D.min()})")

    if latent:
        D = np.linalg.eigvalsh(L_t)
        if D.min() < -1e-8:
            print(f"WARNING: L is not positive semidefinite. Solve to higher accuracy! (min EV is {D.min()})")

    if latent:
        sol = {'Omega': Omega_t, 'Theta': Theta_t, 'L': L_t, 'X': X_t}
    else:
        sol = {'Omega': Omega_t, 'Theta': Theta_t, 'X': X_t}

    if measure:
        info = {'status': status, 'runtime': runtime[:iter_t+1], 'residual': residual[:iter_t+1]}
    else:
        info = {'status': status}

    return sol, info

def ADMM_stopping_criterion(Omega, Omega_t_1, Theta, L, X, S, rho, eps_abs, eps_rel, latent=False):
    # X is inputed as scaled dual variable, this is accounted for by factor rho in e_dual
    if not latent:
        assert np.all(L == 0)

    (p, p) = S.shape


    dim = ((p ** 2 + p) / 2)  # number of elements of off-diagonal matrix
    e_pri = dim * eps_abs + eps_rel * np.maximum(np.linalg.norm(Omega), np.linalg.norm(Theta -L))
    e_dual = dim * eps_abs + eps_rel * rho * np.linalg.norm(X)

    r = np.linalg.norm(Omega - Theta + L)
    s = rho*np.linalg.norm(Omega - Omega_t_1)

    return r,s,e_pri,e_dual

def kkt_stopping_criterion(Omega, Theta, L, X, S, lambda1, latent=False, mu1=None):
    assert Omega.shape == Theta.shape == S.shape
    assert S.shape[0] == S.shape[1]

    if not latent:
        assert np.all(L == 0)

    (p, p) = S.shape

    term1 = np.linalg.norm(Theta - prox_od_1norm(Theta + X, l=lambda1)) / (1 + np.linalg.norm(Theta))

    term2 = np.linalg.norm(Omega - Theta + L) / (1 + np.linalg.norm(Theta))

    eigD, eigQ = np.linalg.eigh(Omega - S - X)
    proxO = phiplus(beta=1, D=eigD, Q=eigQ)
    term3 = np.linalg.norm(Omega - proxO) / (1 + np.linalg.norm(Omega))

    term4 = 0
    if latent:
        eigD, eigQ = np.linalg.eigh(L - X)
        proxL = prox_rank_norm(A=L - X, beta=mu1, D=eigD, Q=eigQ)
        term4 = np.linalg.norm(L - proxL) / (1 + np.linalg.norm(L))

    residual = max(term1, term2, term3, term4)

    return residual


#######################################################
## BLOCK-WISE GRAPHICAL LASSO AFTER WITTEN ET AL.
#######################################################

def block_SGL(S, lambda1, Omega_0, Theta_0=None, X_0=None, rho=1., max_iter=1000, 
              tol=1e-7, rtol=1e-3, stopping_criterion="boyd",
              update_rho=True, verbose=False, measure=False):
    """
    This is a wrapper for solving SGL problems on connected components of the solution and solving each block separately.
    See Witten, Friedman, Simon "New Insights for the Graphical Lasso" for details.

    It solves

    .. math::
       \min_{\Omega, \Theta \in \mathbb{S}^p_{++}} - \log \det \Omega + \mathrm{Tr}(S\Omega) + \lambda \|\Theta\|_{1,od}

       s.t. \quad \Omega = \Theta


    Note:
        * In the original paper the l1-norm is applied as well on the diagonal (here: off-diagonal) which results in a small modification.
        * The returned solution for X is not guaranteed to be identical to the dual variable of the full solution, but can be used as starting point (e.g. in grid search).

    Parameters
    ----------
    S : array (p,p)
        empirical covariance matrix. Needs to be symmetric and positive semidefinite.
    lambda1 : float, positive
        sparsity regularization parameter.
    Omega_0 : array (p,p)
        starting point for the Omega variable. Choose ``np.eye(p)`` if no better starting point is known.
    Theta_0 : array (p,p), optional
        starting point for the Theta variable. If not specified, it is set to the same as Omega_0.
    X_0 : array (p,p), optional
        starting point for the X variable. If not specified, it is set to zero array.
    rho : float, positive, optional
        step size paramater for the augmented Lagrangian in ADMM. The default is 1. Tune this parameter for optimal performance.
    max_iter : int, optional
        maximum number of iterations. The default is 1000.
    tol : float, positive, optional
        tolerance for the primal residual. See "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", Boyd et al. for details.
        The default is 1e-7.
    rtol : float, positive, optional
        tolerance for the dual residual. The default is 1e-4.
    stopping_criterion : str, optional

        * 'boyd': Stopping criterion after Boyd et al.
        * 'kkt': KKT residual is chosen as stopping criterion. This is computationally expensive to compute.

        The default is 'boyd'.
    update_rho : boolean, optional
        Whether the penalty parameter ``rho`` is updated, see Boyd et al. page 20-21 for details. The default is True.
    verbose : boolean, optional
        verbosity of the solver. The default is False.
    measure : boolean, optional
        turn on/off measurements of runtime per iteration. The default is False.

    Returns
    -------
    sol : dict
        contains the solution, i.e. Omega, Theta, X after termination.
    info : dict
        status and measurement information from the solver.
    """
    assert Omega_0.shape == S.shape
    assert S.shape[0] == S.shape[1]
    assert lambda1 > 0

    (p, p) = S.shape

    if Theta_0 is None:
        Theta_0 = Omega_0.copy()
    if X_0 is None:
        X_0 = np.zeros((p, p))

    # compute connected components of S with lambda_1 threshold
    numC, allC = get_connected_components(S, lambda1)

    allOmega = list()
    allTheta = list()
    allX = list()

    for i in range(numC):
        C = allC[i]

        # single node connected components have a closed form solution, see Witten, Friedman, Simon "NEW INSIGHTS FOR THE GRAPHICAL LASSO "
        if len(C) == 1:
            # we use the OFF-DIAGONAL l1-penalty, otherwise it would be 1/(S[C,C]+lambda1)
            closed_sol = 1 / (S[C, C])

            allOmega.append(closed_sol)
            allTheta.append(closed_sol)
            allX.append(np.array([0]))


        # else solve Graphical Lasso for the corresponding block
        else:
            block_S = S[np.ix_(C, C)]
            block_sol, block_info = ADMM_SGL(S=block_S, lambda1=lambda1, Omega_0=Omega_0[np.ix_(C, C)],
                                             Theta_0=Theta_0[np.ix_(C, C)], X_0=X_0[np.ix_(C, C)], tol=tol, rtol=rtol,
                                             stopping_criterion=stopping_criterion, update_rho=update_rho,
                                             rho=rho, max_iter=max_iter, verbose=verbose, measure=measure)

            allOmega.append(block_sol['Omega'])
            allTheta.append(block_sol['Theta'])
            allX.append(block_sol['X'])

    # compute inverse permutation
    per = np.hstack(allC)
    per1 = invert_permutation(per)

    # construct solution by applying inverse permutation indexing
    sol = dict()
    sol['Omega'] = block_diag(*allOmega)[np.ix_(per1, per1)]
    sol['Theta'] = block_diag(*allTheta)[np.ix_(per1, per1)]
    sol['X'] = block_diag(*allX)[np.ix_(per1, per1)]

    return sol


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


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s