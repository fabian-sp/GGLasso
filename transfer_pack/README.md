# GGLasso SLR Transfer Pack

Two modifications to the single (sparse + low-rank, "SLR") graphical lasso ADMM
solver, for integration into GGLasso, plus data to validate them against the R
**SpiecEasi** SLR implementation.

The code under `code/` is a standalone variant of GGLasso's single-GL solver.
The two modifications below are the parts to integrate.

---

## 1. The two modifications

### (a) On/off diagonal penalization in the L1 penalty — `shrink_diag`

- **Where:** `code/solver.py`, `ADMM_single()` — parameter at **line 18**
  (`shrink_diag=False`), implemented at **lines 102–107**.
- **Behaviour:** when `shrink_diag=True`, the empirical covariance `S` is rescaled
  to correlation form `S_ij / sqrt(S_ii · S_jj)` and an entry-wise penalty mask
  `1 / (d_i · d_j)` is applied to `lambda1` (via `lambda1_mask`). Equivalent to
  SpiecEasi's `shrinkDiag=TRUE`. The back-transform to the original scale happens
  after the solve (`code/solver.py` ~line 247).
- **Helper:** `prox_od_1norm(A, l, diag=True)` in `code/helper.py` **line 617** —
  with `diag=True` the diagonal of `A` is preserved (off-diagonal
  soft-thresholding only).
- **Integration target:** `gglasso/solver/single_admm_solver.py` (`ADMM_SGL`);
  prox in `gglasso/solver/ggl_helper.py` (`prox_od_1norm`).

### (b) Explicit rank criterion replacing the continuous `mu1` — `r`

- **Where:** `code/solver.py`, `prox_rank_norm()` — definition at **line 333**,
  rank logic at **lines 339–344**; called in the `L`-update at **lines 174–180**.
- **Behaviour:** instead of thresholding singular values continuously at
  `beta = mu1 / rho` (nuclear-norm prox), passing an integer `r` fixes the rank
  by thresholding at the `r`-th largest eigenvalue (`beta = D[-(r+1)]`). Mirrors
  SpiecEasi's fixed-rank SLR (`src/ADMM.cpp`, the `method='slr', r=...` path).
- **Integration target:** `gglasso/solver/ggl_helper.py` (`prox_rank_norm`),
  invoked from the latent branch of `ADMM_SGL`.
- With `r` set, `mu1` is not required (the latent-branch assertion accepts `r`
  alone). The only combination still needing `mu1` is `stopping_criterion='kkt'`
  with `r` set, which raises an error pointing to `'boyd'`.

`code/stability_selection.py` is the StARS subsampling/instability routine
(SpiecEasi/`huge` style) for selecting `lambda1` along a path.

---

## 2. Files

### `code/` — solver

| File | Contents |
|------|----------|
| `solver.py` | `ADMM_single()` (with `shrink_diag`, `r`), `prox_rank_norm()` |
| `helper.py` | `prox_od_1norm(..., diag=)` and other prox/helpers |
| `stability_selection.py` | StARS `subsample()` + `estimate_instability()` |

`solver.py` imports `phiplus` from `gglasso.solver.ggl_helper` (with an inline
NumPy fallback) and `prox_od_1norm` from `helper.py`.

### `data/inputs/` — solver inputs (empirical covariance)

| File | Shape | Description |
|------|-------|-------------|
| `cov_smoker.csv` | 40×40 (+ index) | empirical covariance, smoker group |
| `cov_non_smoker.csv` | 40×40 (+ index) | empirical covariance, non-smoker group |

### `data/spiec_easi_slr/` — SLR ground-truth (optimal solution, both groups)

| File | Shape | Description |
|------|-------|-------------|
| `theta_smoker.csv`, `theta_non_smoker.csv` | 40×40 (+ index) | sparse precision Θ̂ at the StARS-optimal λ (`method='slr'`, `r=10`) |
| `low_rank_smoker.csv`, `low_rank_non_smoker.csv` | 40×40 (+ index) | low-rank component L (`r=10`) |

StARS-optimal λ index per group: smoker = 10, non-smoker = 8.

### `data/spiec_easi_slr_path/{smoker,non_smoker}/` — full SLR λ-path

Per group: `theta_01..20.csv` (sparse precision), `low_rank_01..20.csv`
(low-rank L), and `lambda_path.csv` (λ value + StARS-optimal flag). Produced by
`regenerate_slr_path.R`. The optimal-index slices equal the files in
`data/spiec_easi_slr/`.

### `data/glasso_path_non_smoker/` — plain glasso (sparse-only) λ-path, non-smoker

| File | Shape | Description |
|------|-------|-------------|
| `sub_icov_1.csv` … `sub_icov_20.csv` | 40×40 (no index) | precision Θ̂ along the 20-step λ path from `method='glasso'` (sparse only, no low-rank), non-smoker group |

Use these to validate the solver's **sparse-only** mode (`latent=False`,
`r=None`), not the low-rank functionality.

### `data/raw/` — count matrices (inputs for `regenerate_slr_path.R`)

| File | Shape | Description |
|------|-------|-------------|
| `counts_smoker.csv`, `counts_non_smoker.csv` | 234 samples × 40 taxa (+ index) | raw counts fed to `spiec.easi` |

**CSV format:** `cov_*`, `theta_*`, `low_rank_*`, and `counts_*` carry a
row-name/index column (read with `index_col=0`); the `sub_icov_*` files have no
index column (40 columns, not 41). All matrices cover 40 taxa. The data are
aggregate covariance/precision matrices from the public American Gut Project.

---

## 3. Validate against the SLR ground-truth (Python)

Run `validate.py` from this directory:

```bash
python validate.py                 # both groups
python validate.py --lambda1 0.05  # custom L1 penalty
python validate.py --group smoker  # one group
```

It loads the `cov_*` inputs, runs

```python
sol, info = ADMM_single(S, lambda1=<λ>, Omega_0=np.eye(p),
                        latent=True, r=10, shrink_diag=True,
                        stopping_criterion="boyd")
```

and reports, per group, Frobenius + relative error of `sol['Theta']` vs.
`theta_*.csv` (with off-diagonal support agreement) and `sol['L']` vs.
`low_rank_*.csv`, plus the recovered `rank(L)`.

- `--lambda1` defaults to an illustrative value. To reproduce the StARS-optimal
  solution, pass the SpiecEasi-selected λ.
- `validate.py` aliases the local `helper` module as `utils.helper` so
  `solver.py`'s imports resolve in this flat folder; under GGLasso these point at
  `gglasso.solver.ggl_helper`.

To validate the **sparse-only** mode, sweep the 20 λ values with
`ADMM_single(S, lambda1=λ_i, ..., latent=False)` on the non-smoker covariance and
compare each estimate against `data/glasso_path_non_smoker/sub_icov_i.csv`.

---

## 4. Regenerate the SLR λ-path (R)

```bash
sbatch regenerate_slr_path.sbatch    # SLURM (StARS sweep — do not run on a login node)
# or: Rscript regenerate_slr_path.R
```

Requires SpiecEasi with `method='slr'`. Reads `data/raw/counts_*.csv`, runs SLR
with `r=10, nlambda=20, lambda.min.ratio=1e-2, sel.criterion='stars',
rep.num=20`, and writes `data/spiec_easi_slr_path/{smoker,non_smoker}/`
(`theta_01..20.csv`, `low_rank_01..20.csv`, `lambda_path.csv`). Uses each group's
own `getOptInd()` for the optimal index.

---

## 5. Source and dependencies

- Code: `Causal_Sparse_Low_Rank_Microbiome_Tutorial/utils/`.
- Data: `Causal_Microbiome_Tutorial/` (`design_AG/`, generated by
  `5_networks_AG/5.1_Networks_compare_AG.Rmd`).
- Pinned: `gglasso==0.2.1`.
