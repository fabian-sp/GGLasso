"""
@author: Fabian Schaipp
"""

import numpy as np
import time
import warnings

from .ggl_helper import prox_sum_Frob, phiplus, prox_rank_norm


def ADMM_FSGL(S, lambda1, M, Omega_0, Theta_0=np.array([]), X_0=np.array([]),
             rho=1., max_iter=1000, tol=1e-7, rtol=1e-4,\
             update_rho=True, verbose=False, measure=False, latent=False, mu1=None):
    """
    This is an ADMM solver for the (Latent variable) Functional Single Graphical Lasso problem (FSGL).
    It solves a SGL problem for the case when each of the ``p`` variables has an ``M``-dimensional functional representation (e.g. Fourier transform).
    Therefore, the input data has the shape ``(p*M,p*M)``.  
    
    If ``latent=False``, this function solves

    .. math::
       \min_{\Omega, \Theta \in \mathbb{S}^{p\cdot M}_{++}} - \log \det \Omega + \mathrm{Tr}(S\Omega) + \lambda_1 \sum_{j\neq l} \|\Theta_{jl}^M\|_{F}

       s.t. \quad \Omega = \Theta

    If ``latent=True``, this function solves

    .. math::
       \min_{\Omega, \Theta, L \in \mathbb{S}^{p\cdot M}_{++}} - \log \det (\Omega) + \mathrm{Tr}(S \Omega) + \lambda_1 \sum_{j\neq l} \|\Theta_{jl}^M\|_{F} + \mu_1 \|L\|_{\star}

       s.t. \quad \Omega = \Theta - L

    Note:
        * Typically, ``sol['Omega']`` is positive definite and ``sol['Theta']`` is sparse.
        * We use scaled ADMM, i.e. X are the scaled (with ``1/rho``) dual variables for the equality constraint.
    Parameters
    ----------
    S : array (p*M,p*M)
        empirical covariance matrix. Needs to be symmetric and positive semidefinite.
    lambda1 : float, positive
        regularization parameter for the Frobenius norm of the MxM subblocks.
    M : int
        Dimension of the functional representation. See "Functional Graphical Models", Qiao et al. for details.
    Omega_0 : array (p*M,p*M)
        starting point for the Omega variable. Choose ``np.eye(p*M)`` if no better starting point is known.
    Theta_0 : array (p*M,p*M), optional
        starting point for the Theta variable. If not specified, it is set to the same as Omega_0.
    X_0 : array (p*M,p*M), optional
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


    if latent:
        assert mu1 is not None
        assert mu1 > 0

    (pM,pM) = S.shape

    assert pM%M == 0
    p = int(pM/M)
    
    if verbose:
        print(f"Derived a Functional SGL problem of dimensionality p={p}.")

    assert rho > 0, "ADMM penalization parameter must be positive."

    # initialize
    Omega_t = Omega_0.copy()

    if len(Theta_0) == 0:
        Theta_0 = Omega_0.copy()
    if len(X_0) == 0:
        X_0 = np.zeros((pM, pM))

    Theta_t = Theta_0.copy()
    L_t = np.zeros((pM, pM))
    X_t = X_0.copy()

    runtime = np.zeros(max_iter)
    residual = np.zeros(max_iter)
    status = ''


    if verbose:
        print("------------ADMM Algorithm for Functional Single Graphical Lasso----------------")
  
        hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s\t%10s"
        out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
        print(hdr_fmt % ("iter", "r_t", "s_t", "eps_pri", "eps_dual", "rho"))
        
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
        Theta_t = prox_sum_Frob(Omega_t + L_t + X_t, M, (1/rho)*lambda1)
        
        # L Update
        if latent:
            C_t = Theta_t - X_t - Omega_t
            eigD1, eigQ1 = np.linalg.eigh(C_t)
            L_t = prox_rank_norm(C_t, mu1/rho, D=eigD1, Q=eigQ1)

        # X Update
        X_t = X_t + Omega_t - Theta_t + L_t

                
        if measure:
            end = time.time()
            runtime[iter_t] = end - start

        # Stopping criterion
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
            print(out_fmt % (iter_t,r_t,s_t,e_pri,e_dual,rho))
        if (r_t <= e_pri) and  (s_t <= e_dual):
            status = 'optimal'
            break



    ##################################################################
    ### MAIN LOOP FINISHED
    ##################################################################
    # retrieve status (partially optimal or max iter)
    if status != 'optimal':
        if (r_t <= e_pri):
                status = 'primal optimal'
        elif (s_t <= e_dual):
            status = 'dual optimal'
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
        warnings.warn(
            f"Theta (Theta - L resp.) is not positive definite. Solve to higher accuracy! (min EV is {D.min()})")

    if latent:
        D = np.linalg.eigvalsh(L_t)
        if D.min() < -1e-8:
            warnings.warn(f"L is not positive semidefinite. Solve to higher accuracy! (min EV is {D.min()})")

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

    (pM,pM) = S.shape


    dim = ((pM ** 2 + pM) / 2)  # number of elements of off-diagonal matrix
    e_pri = dim * eps_abs + eps_rel * np.maximum(np.linalg.norm(Omega), np.linalg.norm(Theta-L))
    e_dual = dim * eps_abs + eps_rel * rho * np.linalg.norm(X)

    r = np.linalg.norm(Omega - Theta + L)
    s = rho*np.linalg.norm(Omega - Omega_t_1)

    return r,s,e_pri,e_dual