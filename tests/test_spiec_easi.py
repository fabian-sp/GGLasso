"""
Validation of the sparse + low-rank single graphical lasso solver
against the SLR estimator from the R package SpiecEasi.

SpiecEasi was run on a synthetic covariance matrix, results and inouts are stored as csv.
"""
import pytest as pt
import pandas as pd
import numpy as np
from pathlib import Path

from gglasso.solver.single_admm_solver import ADMM_SGL

RANK = 5  # fixed rank for L

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "data" / "spiec_easi" / "inputs"
GT = HERE.parent / "data" / "spiec_easi" / "spiec_easi_slr"


def load_matrix(path: str, indexed=True):
    """Load csv matrix as a float ndarray.
    """
    df = pd.read_csv(path, index_col=0 if indexed else None)
    return df.to_numpy(dtype=float)


def compare(name: str, est: np.ndarray, ref: np.ndarray, sparse: bool=True):
    """Print Frobenius / relative error; off-diag support agreement for sparse
    matrices only (it is not meaningful for the dense low-rank component L)."""
    fro = np.linalg.norm(est - ref)
    rel = fro / (np.linalg.norm(ref) + 1e-12)
    line = f"    {name:12s} Frobenius={fro:.2e}  relative={rel:.2e}"
    if sparse:
        p = ref.shape[0]
        offdiag = ~np.eye(p, dtype=bool)
        est_nz = (np.abs(est) > 1e-8) & offdiag
        ref_nz = (np.abs(ref) > 1e-8) & offdiag
        agree = np.mean(est_nz == ref_nz)
        line += f"  off-diag support agreement={agree:.3f}"
    print(line)

    return rel

def count_edges(theta: np.ndarray, edge_threshold: float=1e-3) -> int:
    values = np.abs(theta[np.triu_indices_from(theta, k=1)])
    return int(np.sum(values > edge_threshold))

@pt.mark.parametrize("lmbda", [0.0003, 0.006623])
def test_fixed_lambda(lmbda: float):
    """
    Compares SpiecEasi and GGLasso solutions at fixed (lambda, rank) value.
    """
    lmbda_str = str(lmbda).split(".")[-1]

    cov_path = INPUTS / f"example30_raw_covariance.csv"                         # Input covariance
    theta_path = GT / f"lambda_{lmbda_str}_sparse_precision.csv"                # SpiecEasi sparse solution
    lowrank_path = GT / f"lambda_{lmbda_str}_lowrank.csv"                       # SpiecEasi lowrank solution


    S0 = load_matrix(cov_path)
    p = S0.shape[0]
    print(f"S: {p}x{p}   lambda1={lmbda}   r={RANK} ")

    # call GGLasso with options to match SpiecEasi
    sol, info = ADMM_SGL(
        S0,
        lambda1=lmbda,
        Omega_0=np.eye(p),
        latent=True,
        mu1=RANK,
        stopping_criterion="boyd",
        tol=1e-10,
        rtol=1e-10,
        max_iter=10000,
        verbose=False,
        lambda1_mask=None,
        off_diagonal_l1=False,
        fix_latent_rank=True,
    )
    print(f"    solver status: {info.get('status')}")

    # GGLasso solution
    Theta, L = sol["Theta"], sol["L"]
    print(f"GGLasso solution:    edges(Theta) = {count_edges(Theta)}")
    print(f"                     rank(L) = {np.linalg.matrix_rank(L, tol=1e-6)} ")

    # SpiecEasi solution
    Theta_ref = load_matrix(theta_path)
    L_ref = load_matrix(lowrank_path)
    print(f"SpiecEasi solution:  edges(Theta) = {count_edges(Theta_ref)}")
    print(f"                     rank(L) = {np.linalg.matrix_rank(L_ref, tol=1e-6)} ")

    print("Frobenius distance:")
    rel_dist_theta = compare("Theta", Theta, Theta_ref, sparse=True)
    rel_dist_L = compare("L", L, L_ref, sparse=False)
    print()

    assert rel_dist_theta <= 2e-4
    assert rel_dist_L <= 2e-4
