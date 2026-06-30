#!/usr/bin/env python
"""
One-command validation of the SLR (sparse + low-rank) single graphical lasso
against the R SpiecEasi ground-truth shipped in this transfer pack.

Loads the American Gut empirical covariance inputs, runs the modified solver
with the explicit rank (r=10) and diagonal-penalty switch enabled, and compares
the result against the SpiecEasi SLR outputs.

Usage (from the transfer_pack/ directory):
    python validate.py                      # both groups, default lambda1
    python validate.py --lambda1 0.05       # custom L1 penalty
    python validate.py --group smoker       # one group only

Notes
-----
* The SpiecEasi optimal solution (theta_*, low_rank_*) was selected by StARS at
  a data-dependent lambda. To reproduce it *exactly* you must pass the
  StARS-selected lambda via --lambda1; the default below is illustrative and is
  meant to show the comparison machinery, not to hit the optimum on the nose.
* `solver.py` imports `prox_od_1norm` via `from utils.helper import ...`. In this
  flat hand-off layout we alias the local `helper` module as `utils.helper`
  before importing `solver`. During GGLasso integration these imports get
  rewired to `gglasso.solver.ggl_helper`, so the shim is only for this folder.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
INPUTS = HERE / "data" / "inputs"
GT = HERE / "data" / "spiec_easi_slr"

# --- import shim: make `from utils.helper import prox_od_1norm` resolve ---
sys.path.insert(0, str(CODE))
import helper  # noqa: E402  (transfer_pack/code/helper.py)

_utils = types.ModuleType("utils")
_utils.helper = helper
sys.modules.setdefault("utils", _utils)
sys.modules.setdefault("utils.helper", helper)

from solver import ADMM_single  # noqa: E402  (transfer_pack/code/solver.py)

RANK = 10  # matches spiec.easi(..., method='slr', r=10, ...)


def load_matrix(path, indexed=True):
    """Load a saved R matrix CSV as a float ndarray.

    indexed=True  -> file has a row-name column (cov_*, theta_*, low_rank_*).
    indexed=False -> file written with row.names=FALSE (sub_icov_*).
    """
    df = pd.read_csv(path, index_col=0 if indexed else None)
    return df.to_numpy(dtype=float)


def compare(name, est, ref, sparse=True):
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


def run_group(group, lambda1):
    cov_path = INPUTS / f"cov_{group}.csv"
    theta_path = GT / f"theta_{group}.csv"
    lowrank_path = GT / f"low_rank_{group}.csv"

    S = load_matrix(cov_path)
    p = S.shape[0]
    print(f"[{group}]  S: {p}x{p}   lambda1={lambda1}   r={RANK}   shrink_diag=True")

    # Explicit-rank solve: r fixes the low-rank prox, so mu1 is not needed
    # (the relaxed latent-branch assertion accepts r alone under 'boyd').
    sol, info = ADMM_single(
        S, lambda1=lambda1, Omega_0=np.eye(p),
        latent=True, r=RANK, shrink_diag=True,
        stopping_criterion="boyd",
        tol=1e-7, rtol=1e-4, max_iter=1000, verbose=False,
    )
    print(f"    solver status: {info.get('status')}")

    theta_ref = load_matrix(theta_path)
    lowrank_ref = load_matrix(lowrank_path)
    print("    vs SpiecEasi SLR optimal solution:")
    compare("Theta", sol["Theta"], theta_ref, sparse=True)
    compare("L", sol["L"], lowrank_ref, sparse=False)
    print(f"    recovered rank(L) ~ {np.linalg.matrix_rank(sol['L'], tol=1e-6)} "
          f"(target r={RANK})")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda1", type=float, default=0.1,
                    help="L1 penalty (default 0.1, illustrative). Pass the "
                         "StARS-selected value to match the optimum exactly.")
    ap.add_argument("--group", choices=["smoker", "non_smoker", "both"],
                    default="both", help="which group to validate")
    args = ap.parse_args()

    groups = ["smoker", "non_smoker"] if args.group == "both" else [args.group]
    for g in groups:
        run_group(g, args.lambda1)

    print("Done. Lower Frobenius/relative error and higher support agreement "
          "indicate closer match to the R SpiecEasi SLR result.")


if __name__ == "__main__":
    main()
