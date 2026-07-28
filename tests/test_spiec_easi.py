"""
Validation of the sparse + low-rank single graphical lasso solver
against the R SpiecEasi estimator.

SpiecEasi was run on the American Gut data with explicit rank (r=10) and diagonal-penalty switch enabled,
the results are stored in data/spec_easi (full path and optimal lambda from StARS).


Notes
-----
* The SpiecEasi optimal solution (theta_*, low_rank_*) was selected by StARS at
  a data-dependent lambda. To reproduce it *exactly* you must pass the
  StARS-selected lambda via --lambda1; see data/spiec_easi/spiec_easi_slr_path/{group}/lambda_path.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.helper.basic_linalg import scale_array_by_diagonal

RANK = 10  # matches spiec.easi(..., method='slr', r=10, ...)

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "data" / "spiec_easi" / "inputs"
GT = HERE.parent / "data" / "spiec_easi" / "spiec_easi_slr_path"


def load_matrix(path: str, indexed=True):
    """Load csv matrix as a float ndarray.

    indexed=True  -> file has a row-name column (cov_*, theta_*, low_rank_*).
    indexed=False -> file written with row.names=FALSE (sub_icov_*).
    """
    df = pd.read_csv(path, index_col=0 if indexed else None)
    return df.to_numpy(dtype=float)


def compare(name: str, est: np.ndarray, ref: np.ndarray, sparse: bool=True):
    """Print Frobenius / relative error; off-diag support agreement for sparse
    matrices only (it is not meaningful for the dense low-rank component L)."""
    fro = np.linalg.norm(est - ref)
    rel = fro / (np.linalg.norm(ref) + 1e-12)
    line = f"    {name:12s} Frobenius={fro:.4f}  relative={rel:.4f}"
    if sparse:
        p = ref.shape[0]
        offdiag = ~np.eye(p, dtype=bool)
        est_nz = (np.abs(est) > 1e-8) & offdiag
        ref_nz = (np.abs(ref) > 1e-8) & offdiag
        agree = np.mean(est_nz == ref_nz)
        line += f"  off-diag support agreement={agree:.3f}"
    print(line)


def test_fixed_lambda(group: str, ix: int):
    """
    Compares SpiecEasi and GGLasso solutions at fixed (lambda, rank) value.
    """
    cov_path = INPUTS / f"cov_{group}.csv"                                      # Input covariance
    theta_path = GT / f"{group}" / f"theta_{ix:02}.csv"                         # SpiecEasi sparse ground truth
    lowrank_path = GT / f"{group}" / f"low_rank_{ix:02}.csv"                    # SpiecEasi lowrank ground truth

    lambda_lookup = pd.read_csv(GT / f"{group}" / f"lambda_path.csv", index_col=0)
    lambda1 = lambda_lookup.loc[ix, "lambda"]

    S0 = load_matrix(cov_path)
    p = S0.shape[0]
    print(f"[{group}]  S: {p}x{p}   lambda1={lambda1}   r={RANK} ")

    # scale to correlation matrix
    _scale = np.diag(S0) 
    S = scale_array_by_diagonal(S0)

    lambda1_mask = 1 / np.outer(np.sqrt(_scale), np.sqrt(_scale))

    # Explicit-rank solve: r fixes the low-rank prox, so mu1 is not needed
    # (the relaxed latent-branch assertion accepts r alone under 'boyd').
    sol, info = ADMM_SGL(
        S,
        lambda1=lambda1,
        Omega_0=np.eye(p),
        latent=True,
        mu1=RANK,
        stopping_criterion="boyd",
        tol=1e-9,
        rtol=1e-6,
        max_iter=1000,
        verbose=False,
        lambda1_mask=lambda1_mask,
        off_diagonal_l1=False,
        fix_latent_rank=True,
    )
    print(f"    solver status: {info.get('status')}")

    # rescale to covariance matrix
    Theta, L = sol["Theta"], sol["L"]
    Theta = scale_array_by_diagonal(Theta, d=_scale)
    L = scale_array_by_diagonal(L, d=_scale)
    print(f"GGLasso non-zeros (Theta): {np.count_nonzero(Theta)}")
    print(f"        rank (L) ~ {np.linalg.matrix_rank(L, tol=1e-6)} "
              f"(target r={RANK})")

    # SpiecEasi solutions
    Theta_ref = load_matrix(theta_path)
    L_ref = load_matrix(lowrank_path)
    print(f"SpiecEasi solution non-zeros: {np.count_nonzero(Theta_ref)}")

    print("Frobenius distance:")
    compare("Theta", Theta, Theta_ref, sparse=True)
    compare("L", L, L_ref, sparse=False)
    print()



group = "smoker"
ix = 10
