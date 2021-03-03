"""
author: Fabian Schaipp
"""

import numpy as np
import time
from scipy.sparse.csgraph import connected_components
from scipy.linalg import block_diag

from .ggl_helper import prox_od_1norm, phiplus, prox_rank_norm


def ADMM_stopping_criterion(Omega, Theta, Theta_t_1, L, X, S, tol, latent=False,
                            mu1=None):
    assert Omega.shape == Theta.shape == S.shape
    assert S.shape[0] == S.shape[1]

    if not latent:
        assert np.all(L == 0)

    (p, p) = S.shape

    eps_rel = 1e-3
    eps_abs = tol

    psi = (p ** 2 / 2 + p / 2)  # number of elements of off-diagonal matrix
    e_pri = psi * eps_abs + eps_rel * np.maximum(np.linalg.norm(Omega), np.linalg.norm(Theta))
    e_dual = psi * eps_abs + eps_rel * np.linalg.norm(X)

    r_k = np.linalg.norm(Omega - Theta)
    s_k = np.linalg.norm(Theta - Theta_t_1)

    term4 = 0
    if latent:
        eigD, eigQ = np.linalg.eigh(L - X)
        proxL = prox_rank_norm(A=L - X, beta=mu1, D=eigD, Q=eigQ)
        term4 = np.linalg.norm(L - proxL) / (1 + np.linalg.norm(L))

    status_set = set()
    # primal convergence condition
    if r_k <= e_pri:
        status_set.add("primal optimal")
    # dual convergence condition
    if s_k <= e_dual:
        status_set.add("dual optimal")

    stop_value = max(r_k, s_k, term4)

    return stop_value, status_set


def ADMM_SGL(S, lambda1, Omega_0, Theta_0=np.array([]), X_0=np.array([]), rho=1., max_iter=1000, tol=1e-3,
             verbose=False, measure=False, latent=False, mu1=None):
    """
    This is an ADMM algorithm for solving the Single Graphical Lasso problem

    Omega_0 : start point -- must be specified as a (p,p) array
    S : empirical covariance matrix -- must be specified as a (p,p) array

    latent: boolean to indidate whether low rank term should be estimated
    mu1: low rank penalty paramater, if latent=True


    In the code, X are the SCALED (with 1/rho) dual variables, for the KKT stop criterion they have to be unscaled (i.e. take rho*X) again!
    """
    assert Omega_0.shape == S.shape
    assert S.shape[0] == S.shape[1]
    assert lambda1 > 0

    if latent:
        assert mu1 is not None
        assert mu1 > 0

    (p, p) = S.shape

    assert rho > 0, "ADMM penalization parameter must be positive."

    # initialize
    # status = {0: "not optimal", 1: "primal optimal", 2: "dual optimal"}
    status = "not optimal"
    Omega_t = Omega_0.copy()
    if len(Theta_0) == 0:
        Theta_0 = Omega_0.copy()
    if len(X_0) == 0:
        X_0 = np.zeros((p, p))

    Theta_t = Theta_0.copy()
    Theta_t_1 = Theta_0.copy()
    L_t = np.zeros((p, p))
    X_t = X_0.copy()
    eta_A = (1, set())

    runtime = np.zeros(max_iter)
    residual = np.zeros(max_iter)

    for iter_t in np.arange(max_iter):
        if measure:
            start = time.time()

        if iter_t > 0:
            eta_A = ADMM_stopping_criterion(Omega_t, Theta_t, Theta_t_1, L_t, rho * X_t, S, tol, latent, mu1)
            residual[iter_t] = eta_A[0]

        if len(eta_A[1]) > 1:
            status = 'primal and dual optimal'
            break

        if verbose:
            print(f"------------Iteration {iter_t} of the ADMM Algorithm----------------")

        # Omega Update
        W_t = Theta_t - L_t - X_t - (1 / rho) * S
        eigD, eigQ = np.linalg.eigh(W_t)
        Omega_t = phiplus(W_t, beta=1 / rho, D=eigD, Q=eigQ)

        # Theta Update
        Theta_t_1 = Theta_t.copy()
        Theta_t = prox_od_1norm(Omega_t + L_t + X_t, (1 / rho) * lambda1)

        # L Update
        if latent:
            C_t = Theta_t - X_t - Omega_t
            # C_t = (C_t.T + C_t)/2
            eigD, eigQ = np.linalg.eigh(C_t)
            L_t = prox_rank_norm(C_t, mu1 / rho, D=eigD, Q=eigQ)

        # X Update
        X_t = X_t + Omega_t - Theta_t + L_t

        if measure:
            end = time.time()
            runtime[iter_t] = end - start

        if verbose:
            print(f"Current accuracy: ", eta_A[0])

    if len(eta_A[1]) == 1:
        print(f"ADMM is only {eta_A[1]}")
        print(f"Try to change the tolerance value {tol}")

    if eta_A[0] > tol:
        status = 'max iterations reached'
    print(f"ADMM terminated after {iter_t} iterations with accuracy {eta_A[0]}")
    print(f"ADMM status: {status}")

    assert abs((Omega_t).T - Omega_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs((Theta_t).T - Theta_t).max() <= 1e-5, "Solution is not symmetric"
    assert abs((L_t).T - L_t).max() <= 1e-5, "Solution is not symmetric"

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
        info = {'status': status, 'runtime': runtime[:iter_t], 'residual': residual[1:iter_t + 1]}
    else:
        info = {'status': status}

    return sol, info


#######################################################
## BLOCK-WISE GRAPHICAL LASSO AFTER WITTEN ET AL.
#######################################################

def block_SGL(S, lambda1, Omega_0, tol=1e-5, Theta_0=None, X_0=None, rho=1., max_iter=1000, verbose=False,
              measure=False):
    """
    Parameters
    ----------
    S : (p,p) array
        Empirical covariance matrix. Should be symmetric and semipositive definite.
    lambda1 : float
        Positive l1-regularization parameter.
    Omega_0 : array
        Starting point for solver. Use np.eye(p) if no prior knowledge.
    tol : float, optional
        Tolerance for the ADMM algorithm on each block. The default is 1e-5.
    Theta_0 : array, optional
        Starting point for solver (for Theta variable).
    X_0 : array
        Starting point for solver (for dual variable).
    rho : float, optional
        ADMM penalty parameter. The default is 1..
    max_iter : int, optional
        Maximum number of iterations for the ADMM algorithm on each block. The default is 1000.

    verbose : boolean, optional
        ADMM prints information. The default is False.
    measure : boolean, optional
        Measure runtime and objective at each iter of ADMM. The default is False.

    Returns
    -------
    sol2 : (p,p) array
        Solution Theta to the Graphical Lasso problem.

    This function solves the Single Graphical Lasso problem

    min -log det(Z) + tr(S.T@Z) + lambda_1 * ||Z||_1,od

    by finding connected components of the solution and solving each block separately, according to Witten, Friedman, Simon "NEW INSIGHTS FOR THE GRAPHICAL LASSO"
    where ||Z||_1,od is the off-diagonal l1-norm.


    NOTE:
        -in the original paper the l1-norm is also used on the diagonal which results in a small modification.
        -the returned solution for X is not guaranteed to be identical to the dual variable of the full solution, but can be used as starting point (e.g. in grid search)
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
            block_sol, block_info = ADMM_SGL(S=block_S, lambda1=lambda1, eps_admm=tol, Omega_0=Omega_0[np.ix_(C, C)], \
                                             Theta_0=Theta_0[np.ix_(C, C)], X_0=X_0[np.ix_(C, C)], \
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


# def ADMM_stopping_criterion(Omega, Theta, L, X, S, lambda1, latent=False, mu1=None):
#     assert Omega.shape == Theta.shape == S.shape
#     assert S.shape[0] == S.shape[1]
#
#     if not latent:
#         assert np.all(L == 0)
#
#     (p, p) = S.shape
#
#     term1 = np.linalg.norm(Theta - prox_od_1norm(Theta + X, l=lambda1)) / (1 + np.linalg.norm(Theta))
#
#     term2 = np.linalg.norm(Omega - Theta + L) / (1 + np.linalg.norm(Theta))
#
#     eigD, eigQ = np.linalg.eigh(Omega - S - X)
#     proxO = phiplus(beta=1, D=eigD, Q=eigQ)
#     term3 = np.linalg.norm(Omega - proxO) / (1 + np.linalg.norm(Omega))
#
#     term4 = 0
#     if latent:
#         eigD, eigQ = np.linalg.eigh(L - X)
#         proxL = prox_rank_norm(A=L - X, beta=mu1, D=eigD, Q=eigQ)
#         term4 = np.linalg.norm(L - proxL) / (1 + np.linalg.norm(L))
#
#     return max(term1, term2, term3, term4)