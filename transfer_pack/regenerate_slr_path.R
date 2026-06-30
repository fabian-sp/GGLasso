#!/usr/bin/env Rscript
# Regenerate the FULL SpiecEasi sparse+low-rank (SLR) lambda path for both
# groups of the American Gut smoking example, to serve as validation
# ground-truth for the Python solver.
#
# The transfer pack ships only the SLR *optimal* solution (theta_*, low_rank_*).
# This script reproduces the entire 20-step path: for every lambda it writes the
# sparse precision (est$icov[[i]]) and the low-rank component (est$resid[[i]]),
# for smoker and non-smoker, plus the lambda values and StARS-optimal index.
#
# Inputs (shipped in the pack):
#   data/raw/counts_smoker.csv      (234 samples x 40 taxa, raw counts)
#   data/raw/counts_non_smoker.csv
# These are net_W$countMat1 / countMat2 from the tutorial (NetCoMi result),
# i.e. exactly the matrices fed to spiec.easi in 5.1_Networks_compare_AG.Rmd.
#
# Usage (from the transfer_pack/ directory):
#   Rscript regenerate_slr_path.R
#
# Requires: SpiecEasi (with the SLR / method='slr' support, branch 'lowrank').
#
# NOTE on a bug in the original tutorial: line 311 of the .Rmd extracted the
# non-smoker optimal precision with the SMOKER's optimal index
# (se_1_slr$est$icov[[getOptInd(se_0_slr)]]). This script uses each group's own
# getOptInd(), which is the correct behaviour.

suppressMessages(library(SpiecEasi))

HERE     <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
RAW      <- file.path(HERE, "data", "raw")
OUT_ROOT <- file.path(HERE, "data", "spiec_easi_slr_path")

# SLR parameters — identical to the tutorial (5.1_Networks_compare_AG.Rmd L275-277)
RANK            <- 10
NLAMBDA         <- 20
LAMBDA_MIN_RATIO <- 1e-2
REP_NUM         <- 20

read_counts <- function(path) {
  # samples x taxa, first column = sample IDs
  as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
}

run_group <- function(label, counts_csv) {
  cat(sprintf("\n=== %s ===\n", label))
  X <- read_counts(counts_csv)
  cat(sprintf("counts: %d samples x %d taxa\n", nrow(X), ncol(X)))

  se <- spiec.easi(X, method = 'slr', r = RANK,
                   lambda.min.ratio = LAMBDA_MIN_RATIO, nlambda = NLAMBDA,
                   sel.criterion = 'stars', beta = 0,
                   pulsar.params = list(rep.num = REP_NUM, ncores = 1))

  out <- file.path(OUT_ROOT, label)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  n <- length(se$est$icov)
  opt <- getOptInd(se)
  lambdas <- se$lambda

  for (i in seq_len(n)) {
    icov  <- as.matrix(se$est$icov[[i]])    # sparse precision Theta
    resid <- as.matrix(se$est$resid[[i]])   # low-rank component L
    write.csv(icov,  file.path(out, sprintf("theta_%02d.csv", i)),    row.names = TRUE)
    write.csv(resid, file.path(out, sprintf("low_rank_%02d.csv", i)), row.names = TRUE)
  }

  # path metadata: lambda value + StARS instability + optimal flag
  meta <- data.frame(
    index     = seq_len(n),
    lambda    = lambdas,
    stars_summary = if (!is.null(se$select$stars$summary)) se$select$stars$summary else NA,
    is_optimal = seq_len(n) == opt
  )
  write.csv(meta, file.path(out, "lambda_path.csv"), row.names = FALSE)

  cat(sprintf("wrote %d theta_* + %d low_rank_* to %s\n", n, n, out))
  cat(sprintf("StARS optimal index = %d  (lambda = %.5g)\n", opt, lambdas[opt]))
}

dir.create(OUT_ROOT, recursive = TRUE, showWarnings = FALSE)
run_group("smoker",     file.path(RAW, "counts_smoker.csv"))
run_group("non_smoker", file.path(RAW, "counts_non_smoker.csv"))
cat("\nDone. SLR path written under data/spiec_easi_slr_path/{smoker,non_smoker}/\n")
