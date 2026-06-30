# GGLasso SLR Transfer Pack

Hand-off from **Oleg Vlasovets** to **Fabian Schaipp** for integrating two
modifications to the single (sparse + low-rank, "SLR") graphical lasso ADMM
solver into GGLasso, plus the data needed to validate them against the R
**SpiecEasi** SLR implementation.

This is a **drop-in reference folder**, not a patch against the package source.
The code under `code/` is a standalone variant of GGLasso's single-GL solver;
the two modifications below are the parts intended for integration.

---

## 1. The two modifications

### (a) On/off diagonal penalization in the L1 penalty — `shrink_diag`

- **Where:** `code/solver.py`, `ADMM_single()` — parameter declared at **line 18**
  (`shrink_diag=False`), implemented at **lines 102–107**.
- **What it does:** when `shrink_diag=True`, the empirical covariance `S` is
  rescaled to correlation form `S_ij / sqrt(S_ii · S_jj)` and an entry-wise
  effective penalty mask `1 / (d_i · d_j)` is applied to `lambda1`
  (via `lambda1_mask`). This reproduces SpiecEasi's `shrinkDiag=TRUE` behaviour
  in its C++ ADMM. The back-transform to the original scale happens after the
  solve (`code/solver.py` ~line 247).
- **Helper dependency:** `prox_od_1norm(A, l, diag=True)` in `code/helper.py`
  **line 617** — when `diag=True` the diagonal of `A` is preserved (off-diagonal
  soft-thresholding only).
- **GGLasso integration target:** `gglasso/solver/single_admm_solver.py`
  (`ADMM_SGL`); the prox lives in `gglasso/solver/ggl_helper.py`
  (`prox_od_1norm`).

### (b) Explicit rank criterion replacing the continuous `mu1` — `r`

- **Where:** `code/solver.py`, `prox_rank_norm()` — definition at **line 333**,
  rank logic at **lines 339–344**; called in the `L`-update at **lines 174–180**.
- **What it does:** instead of thresholding singular values continuously at
  `beta = mu1 / rho` (nuclear-norm prox), passing an integer `r` fixes the rank
  directly by thresholding at the `r`-th largest eigenvalue
  (`beta = D[-(r+1)]`), zeroing everything below. This mirrors SpiecEasi's
  fixed-rank SLR (`src/ADMM.cpp`, the `method='slr', r=...` path).
- **GGLasso integration target:** `gglasso/solver/ggl_helper.py`
  (`prox_rank_norm`), invoked from the latent branch of `ADMM_SGL`.

### Stability selection

`code/stability_selection.py` is the StARS subsampling/instability routine
(SpiecEasi/`huge` style) used to select `lambda1` along the path. Included so
the validation recipe below is reproducible end-to-end.

---

## 2. Files

### `code/` — standalone solver variant (copied verbatim)

| File | Contents |
|------|----------|
| `solver.py` | `ADMM_single()` (with `shrink_diag`, `r`), `prox_rank_norm()` |
| `helper.py` | `prox_od_1norm(..., diag=)` and other prox/helpers |
| `stability_selection.py` | StARS `subsample()` + `estimate_instability()` |

`solver.py` imports `phiplus` from `gglasso.solver.ggl_helper` (with an inline
NumPy fallback if gglasso's numba dependency is unavailable) and
`prox_od_1norm` from `helper.py`.

### `data/inputs/` — solver inputs (empirical covariance)

| File | Shape | Notes |
|------|-------|-------|
| `cov_smoker.csv` | 40×40 (+ index) | empirical covariance, smoker group |
| `cov_non_smoker.csv` | 40×40 (+ index) | empirical covariance, non-smoker group |

### `data/spiec_easi_slr/` — R SpiecEasi SLR outputs (validation ground-truth)

| File | Shape | Role |
|------|-------|------|
| `theta_smoker.csv`, `theta_non_smoker.csv` | 40×40 (+ index) | sparse precision Θ̂ at the StARS-optimal λ |
| `low_rank_smoker.csv`, `low_rank_non_smoker.csv` | 40×40 (+ index) | low-rank component L (rank `r=10`) |
| `sub_icov_1.csv` … `sub_icov_20.csv` | 40×40 (**no index**) | precision matrices along the 20-step λ path |

**Format caveat:** `cov_*`, `theta_*`, `low_rank_*` carry a row-name/index
column (`read.csv(..., index_col=0)`); the `sub_icov_*` path files were written
with `row.names=FALSE`, so they have **no index column** (40 columns, not 41).

All matrices cover **40 taxa** after filtering. The data are aggregate
covariance/precision matrices derived from the public American Gut Project — no
individual-level data.

---

## 3. How the validation data was generated (R)

From `Causal_Microbiome_Tutorial/5_networks_AG/5.1_Networks_compare_AG.Rmd`
(lines 275–336):

```r
se_slr <- spiec.easi(countMat, method = 'slr', r = 10,
                     lambda.min.ratio = 1e-2, nlambda = 20,
                     sel.criterion = 'stars', beta = 0,
                     pulsar.params = list(rep.num = 20, ncores = 1))
# optimal solution
S_inv <- se_slr$est$icov[[getOptInd(se_slr)]]   # -> theta_*.csv
L     <- se_slr$est$resid[[getOptInd(se_slr)]]  # -> low_rank_*.csv
# full path: se_slr$est$icov[[i]] for i in 1..20 -> sub_icov_i.csv
```

Key parameters: **`r = 10`**, **`nlambda = 20`**, `lambda.min.ratio = 1e-2`,
StARS selection, `rep.num = 20`.

---

## 4. Suggested validation recipe (Python)

A runnable harness is included: **`validate.py`** (run from this directory).

```bash
python validate.py                 # both groups, illustrative lambda1
python validate.py --lambda1 0.05  # custom L1 penalty
python validate.py --group smoker  # one group
```

It loads the `cov_*` inputs, runs

```python
sol, info = ADMM_single(S, lambda1=<λ>, Omega_0=np.eye(p),
                        latent=True, r=10, mu1=1.0, shrink_diag=True,
                        stopping_criterion="boyd")
```

and reports, per group, Frobenius + relative error of `sol['Theta']` vs.
`theta_*.csv` (with off-diagonal support agreement) and of `sol['L']` vs.
`low_rank_*.csv`, plus the recovered `rank(L)`.

Notes:
- `mu1` must be positive to pass the solver's latent-branch assertion, but it is
  **inert** when `r` is set (the L-update uses the fixed-rank threshold) under
  `boyd` stopping.
- The default `--lambda1` is illustrative; to reproduce the StARS-optimal
  solution exactly, pass the SpiecEasi-selected λ.
- To check the path, sweep the 20 λ values and compare each precision estimate
  against the corresponding `sub_icov_i.csv`; selection should land on the same
  optimal index StARS picked in R.

`validate.py` aliases the local `helper` module as `utils.helper` so the
verbatim `solver.py` imports resolve in this flat folder; during GGLasso
integration these imports point at `gglasso.solver.ggl_helper` instead.

---

## 5. Provenance & open question

- **Source repos:**
  - Code: `Causal_Sparse_Low_Rank_Microbiome_Tutorial/utils/` (the `shrink_diag`
    switch was added in commit `b7b2219`; the `r` criterion pre-dates it).
  - Data: `Causal_Microbiome_Tutorial/design_AG/` (branch `main`).
- **Pinned dependency:** `gglasso==0.2.1`.
- **OPEN QUESTION for Fabian/Oleg:** the `sub_icov_*` λ-path series — confirm
  which group (smoker vs. non-smoker) it corresponds to before relying on it.
  The generating loop in the `.Rmd` is commented out and references
  `se_1$est$icov`, which points to the **non-smoker** SLR fit (`countMat2`), but
  this should be verified against the live R objects.
