"""
author: Fabian Schaipp
"""

import numpy as np
import time
import warnings

from ..helper.basic_linalg import trp, Gdot
from .ggl_helper import prox_p, phiplus, prox_rank_norm, f, P_val


def ADMM_MGL(S, lambda1, lambda2, reg , Omega_0 , \
             Theta_0 = np.array([]), X_0 = np.array([]), n_samples = None, \
             tol = 1e-5 , rtol = 1e-4, stopping_criterion = 'boyd', update_rho = True,\
             rho= 1., max_iter = 1000, verbose = False, measure = False, latent = False, mu1 = None):
    """
    This is an ADMM solver for the (Latent variable) Multiple Graphical Lasso problem (MGL). It jointly estimates K precision matrices of shape (p,p).

    If ``latent=False``, this function solves
    
    .. math::
       \min_{\Omega, \Theta} \sum_{k=1}^{K} (-\log\det(\Omega^{(k)}) + \mathrm{Tr}(S^{(k)} \Omega^{(k)}) ) + \mathcal{P}(\Theta)
       
       s.t. \quad \Omega^{(k)} = \Theta^{(k)} \quad k=1,\dots,K
       
    Here, :math:`\mathcal{P}` is a regularization function which depends on the application. Group Graphical Lasso (GGL) or Fused Graphical Lasso (FGL) is implemented.        
    If ``latent=True``, this function solves
    
    .. math::
       \min_{\Omega, \Theta, L}\quad \sum_{k=1}^{K} (-\log\det(\Omega^{(k)}) + \mathrm{Tr}(S^{(k)},\Omega^{(k)}) ) + \mathcal{P}(\Theta) +\sum_{k=1}^{K} \mu_{1,k} \|L^{(k)}\|_{\star}
       
       s.t. \quad \Omega^{(k)} = \Theta^{(k)} - L^{(k)} \quad k=1,\dots,K
    
    Note:    
        * Typically, ``sol['Omega']`` is positive definite and ``sol['Theta']`` is sparse.
        * We use scaled ADMM, i.e. X are the scaled (with 1/rho) dual variables for the equality constraint. 

    Parameters
    ----------
    S : array (K,p,p)
        empirical covariance matrices, i.e. S[k,:,:] contains the empirical cov. matrix of the k-th instance. 
        Each S[k,:,:] needs to be symmetric and positive semidefinite.
    lambda1 : float, positive
        sparsity regularization parameter.
    lambda2 : float, positive
        group sparsity/ total variation regularization parameter.
    reg : str
        choose either
        
        * 'GGL': Group Graphical Lasso
        * 'FGL': Fused Graphical Lasso
        
    Omega_0 : array (K,p,p)
        starting point for the Omega variable. Use get_K_identity(K, p) from ``gglasso.helper.utils`` if no better starting point is known.
    Theta_0 : array (p,p), optional
        starting point for the Theta variable. If not specified, it is set to the same as Omega_0.
    X_0 : array (p,p), optional
        starting point for the X variable. If not specified, it is set to zero array.
    n_samples : int or array of shape(K,), optional
        neg. log-likelihood is sometimes weighted with the sample size. If this is desired, specify sample size of each instance.
        Default is no weighting, i.e. n_samples = 1.
    rho : float, positive, optional
        step size paramater for the augmented Lagrangian in ADMM. The default is 1. Tune this parameter for optimal performance.
    max_iter : int, optional
        maximum number of iterations. The default is 1000.
    tol : float, positive, optional
        tolerance for the primal residual. See 'Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers', Boyd et al. for details.
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
        Solve the MGL problem with or without latent variables (see above for the exact formulations).
        The default is False.
    mu1 : float or array of shape(K,), positive, optional
        low-rank regularization parameter, possibly different for each instance k=1,..,K. Only needs to be specified if latent=True.

    Returns
    -------
    sol : dict
        contains the solution, i.e. Omega, Theta, X (and L if ``latent=True``) after termination. All arrays are of shape (K,p,p).
    info : dict
        status and measurement information from the solver.

    """
    assert Omega_0.shape == S.shape
    assert S.shape[1] == S.shape[2]
    assert reg in ['GGL', 'FGL']
    assert min(lambda1, lambda2) > 0
        
    (K,p,p) = S.shape
    
    assert rho > 0, "ADMM penalization parameter must be positive."
    
    if latent:
        if type(mu1) == np.float64 or type(mu1) == float:
             mu1 = mu1*np.ones(K)
            
        assert mu1 is not None
        assert np.all(mu1 > 0)
    
    
    # n_samples None --> set them all to 1
    # n_samples integer --> all instances have same number of samples
    # else --> n_samples should be array with sample sizes
    if n_samples == None:
        nk = np.ones((K,1,1))
    elif type(n_samples) == int:
        nk = n_samples * np.ones((K,1,1)) 
    else:
        assert len(nk) == K
        nk = n_samples.reshape(K,1,1)
    
    # initialize 
    Omega_t = Omega_0.copy()
    if len(Theta_0) == 0:
        Theta_0 = Omega_0.copy()
    if len(X_0) == 0:
        X_0 = np.zeros((K,p,p))

    Theta_t = Theta_0.copy()
    L_t = np.zeros((K,p,p))
    X_t = X_0.copy()
     
    runtime = np.zeros(max_iter)
    residual = np.zeros(max_iter)
    objective = np.zeros(max_iter)
    status = ''
    
    if verbose:
        print("------------ADMM Algorithm for Multiple Graphical Lasso----------------")

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
        Omega_t_1 = Omega_t.copy()
        W_t = Theta_t - L_t - X_t - (nk/rho) * S
        eigD, eigQ = np.linalg.eigh(W_t)
        
        for k in np.arange(K):
            Omega_t[k,:,:] = phiplus(beta = nk[k,0,0]/rho, D = eigD[k,:], Q = eigQ[k,:,:])
        
        # Theta Update
        Theta_t = prox_p(Omega_t + L_t + X_t, (1/rho)*lambda1, (1/rho)*lambda2, reg)
        
        #L Update
        if latent:
            C_t = Theta_t - X_t - Omega_t
            eigD, eigQ = np.linalg.eigh(C_t)
            for k in np.arange(K):
                L_t[k] = prox_rank_norm(C_t[k,:,:], mu1[k]/rho, D = eigD[k,:], Q = eigQ[k,:,:])
                
        # X Update
        X_t += Omega_t - Theta_t + L_t
        
        
        if measure:
            end = time.time()
            runtime[iter_t] = end-start
            objective[iter_t] = f(Omega_t, S) + P_val(Theta_t, lambda1, lambda2, reg)
        
        # Stopping condition
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
            eta_A = kkt_stopping_criterion(Omega_t, Theta_t, L_t, rho*X_t, S , lambda1, lambda2, nk, reg, latent, mu1)
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
    if abs(trp(Omega_t)- Omega_t).max() > 1e-5:
        warnings.warn(f"Omega variable is not symmetric, largest deviation is {abs(trp(Omega_t)- Omega_t).max()}.")
    
    if abs(trp(Theta_t)- Theta_t).max() > 1e-5:
        warnings.warn(f"Theta variable is not symmetric, largest deviation is {abs(trp(Theta_t)- Theta_t).max()}.")
    
    if abs(trp(L_t)- L_t).max() > 1e-5:
        warnings.warn(f"L variable is not symmetric, largest deviation is {abs(trp(L_t)- L_t).max()}.")

    ### CHECK FOR POSDEF
    D = np.linalg.eigvalsh(Theta_t - L_t)
    if D.min() <= 0:
        print("WARNING: Theta (Theta - L resp.) is not positive definite. Solve to higher accuracy!")
    
    if latent:
        D = np.linalg.eigvalsh(L_t)
        if D.min() < -1e-5:
            print("WARNING: L is not positive semidefinite. Solve to higher accuracy!")
        
    sol = {'Omega': Omega_t, 'Theta': Theta_t, 'L': L_t, 'X': X_t}
    if measure:
        info = {'status': status , 'runtime': runtime[:iter_t+1], 'residual': residual[:iter_t+1], 'objective': objective[:iter_t+1]}
    else:
        info = {'status': status}
               
    return sol, info


def ADMM_stopping_criterion(Omega, Omega_t_1, Theta, L, X, S, rho, eps_abs, eps_rel, latent=False):
    # X is inputed as scaled dual variable, this is accounted for by factor rho in e_dual
    
    if not latent:
        assert np.all(L == 0)

    (K,p,p) = S.shape

    dim = K*((p ** 2 + p) / 2)  # number of elements of off-diagonal matrix
    e_pri = dim * eps_abs + eps_rel * np.maximum(np.linalg.norm(Omega), np.linalg.norm(Theta -L))
    e_dual = dim * eps_abs + eps_rel * rho * np.linalg.norm(X)

    r = np.linalg.norm(Omega - Theta + L)
    s = rho*np.linalg.norm(Omega - Omega_t_1)

    return r,s,e_pri,e_dual

def kkt_stopping_criterion(Omega, Theta, L, X, S , lambda1, lambda2, nk, reg, latent = False, mu1 = None):
    
    assert Omega.shape == Theta.shape == S.shape
    assert S.shape[1] == S.shape[2]
    
    if not latent:
        assert np.all(L == 0)
    
    (K,p,p) = S.shape
    
    term1 = np.linalg.norm(Theta - prox_p(Theta + X , l1 = lambda1, l2= lambda2, reg = reg)) / (1 + np.linalg.norm(Theta))
    
    term2 = np.linalg.norm(Theta - Omega - L) / (1 + np.linalg.norm(Theta))
    
    proxK = np.zeros((K,p,p))
    eigD, eigQ = np.linalg.eigh(Omega - nk*S - X)
    for k in np.arange(K):       
        proxK[k,:,:] = phiplus(beta = nk[k,0,0], D = eigD[k,:], Q = eigQ[k,:,:])
     
    term3 = np.linalg.norm(Omega - proxK) / (1 + np.linalg.norm(Omega))
    
    if latent:
        proxL = np.zeros((K,p,p))
        eigD, eigQ = np.linalg.eigh(L - X)
        for k in np.arange(K):
            proxL[k,:,:] = prox_rank_norm(L[k,:,:] - X[k,:,:], beta = mu1[k], D = eigD[k,:], Q = eigQ[k,:,:])
        
        term4 = np.linalg.norm(L - proxL) / (1 + np.linalg.norm(L))
    else:
        term4 = 0
        
    return max(term1, term2, term3, term4)

    