#!/usr/bin/env Rscript
# Study C: fit the nuisances once per replicate, then run every R comparator off
# those same fits.
#
#   Rscript R/study_c_nuisances.R --dir results/study_c/n500 [--reps 500]
#
# Two jobs in one pass, and the pairing is the reason they are not split:
#
#   1. Export the fitted nuisances as full (n x K) counterfactual cumulative
#      hazards, so PyTMLE and concrete can be injected with byte-identical
#      inputs (tier 1). Exporting coefficients instead would make the comparison
#      depend on matching R's and scikit-survival's centring conventions -- a
#      silent mismatch there is exactly what this study exists to rule out.
#   2. Run the R estimators on the *same* fitted objects: riskRegression::ate in
#      its three flavours and the conventional cause-specific Cox (tier 2), plus
#
# A conventional cause-specific hazard ratio is not on the risk-difference scale
# and must never be scored for bias against the RD truth. It is emitted with

suppressMessages({
  library(survival)
  library(riskRegression)
  library(prodlim)
  library(data.table)
  library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[[i + 1]] }
DIR      <- getarg("--dir", "results/study_c/n500")
REPS     <- suppressWarnings(as.integer(getarg("--reps", NA)))
# Replicate range, so several processes can share one directory. R is
# single-threaded here and the per-replicate cost runs to ~95 s at n = 2000, so
# without this a full cell is a wall-clock day. Each shard writes its own
# estimates file; `--shard` names it.
FROM     <- suppressWarnings(as.integer(getarg("--from", NA)))
TO       <- suppressWarnings(as.integer(getarg("--to", NA)))
SHARD    <- getarg("--shard", NA)

taus <- as.numeric(read_parquet(file.path(DIR, "taus.parquet"))$time)
data_files <- sort(list.files(DIR, pattern = "^rep[0-9]+_data\\.parquet$",
                              full.names = TRUE))
if (!is.na(REPS)) data_files <- head(data_files, REPS)
if (!is.na(FROM) || !is.na(TO)) {
  idx <- as.integer(sub("^rep([0-9]+)_data\\.parquet$", "\\1", basename(data_files)))
  keep <- rep(TRUE, length(idx))
  if (!is.na(FROM)) keep <- keep & idx >= FROM
  if (!is.na(TO))   keep <- keep & idx <  TO
  data_files <- data_files[keep]
}
if (!length(data_files)) stop("no replicate data found in ", DIR)

FML_OUT  <- Hist(event_time, event_indicator) ~ A + d2 + d3 + w_cont
FML_CENS <- Surv(event_time, event_indicator == 0) ~ A + d2 + d3 + w_cont
FML_TRT  <- A ~ d2 + d3 + w_cont

# `ate` requires the treatment variable to be a factor, and `predictCox` needs
# the same factor levels in newdata, so the coding is fixed once here.
A_LEVELS <- c("0", "1")

rows <- list(); n_fail <- 0L; first_err <- NULL

for (path in data_files) {
  i <- as.integer(sub("^rep([0-9]+)_data\\.parquet$", "\\1", basename(path)))
  res <- tryCatch({
    df <- as.data.table(read_parquet(path))
    df[, A := factor(as.character(group), levels = A_LEVELS)]
    n <- nrow(df)
    grid <- sort(unique(df$event_time))

    d1 <- copy(df)[, A := factor("1", levels = A_LEVELS)]
    d0 <- copy(df)[, A := factor("0", levels = A_LEVELS)]

    # --- the shared fits -------------------------------------------------
    csc  <- CSC(FML_OUT, data = df)
    cens <- coxph(FML_CENS, data = df, x = TRUE, y = TRUE)
    trt  <- glm(FML_TRT, data = df, family = binomial)

    ch <- function(fit, nd) predictCox(fit, newdata = nd, times = grid,
                                       type = "cumhazard")$cumhazard
    stub <- file.path(DIR, sprintf("rep%03d", i))
    write_parquet(data.table(time = grid), paste0(stub, "_grid.parquet"))
    write_parquet(data.table(ps1 = as.numeric(predict(trt, type = "response"))),
                  paste0(stub, "_ps.parquet"))
    mats <- list(H1_1 = ch(csc$models[[1]], d1), H1_0 = ch(csc$models[[1]], d0),
                 H2_1 = ch(csc$models[[2]], d1), H2_0 = ch(csc$models[[2]], d0),
                 HC_1 = ch(cens, d1),            HC_0 = ch(cens, d0))
    stopifnot(all(vapply(mats, function(m) all(dim(m) == c(n, length(grid))), TRUE)))
    for (nm in names(mats)) write_parquet(as.data.frame(mats[[nm]]),
                                          paste0(stub, "_", nm, ".parquet"))

    out <- list()

    # --- tier 2: riskRegression::ate, three estimators off the same fits ---
    # `ate` takes one cause per call (`cause = 1:2` fails inside ate_initArgs),
    # so the loop is over causes as well and the timing is summed across them to
    # stay comparable with the single-call implementations.
    for (est in c("GFORMULA", "IPTW", "AIPTW")) {
      el <- 0.0; got <- list()
      for (j in 1:2) {
        t0 <- Sys.time()
        a <- ate(event = csc, treatment = trt, censor = cens, data = df,
                 estimator = est, times = taus, cause = j, se = TRUE,
                 verbose = FALSE, B = 0)
        el <- el + as.numeric(Sys.time() - t0, units = "secs")
        dl <- as.data.table(a$diffRisk)
        got[[j]] <- data.table(event = j, time = as.numeric(dl$time),
                               est = as.numeric(dl$estimate),
                               se = as.numeric(dl$se),
                               ci_lo = as.numeric(dl$lower),
                               ci_hi = as.numeric(dl$upper))
      }
      g <- rbindlist(got)
      out[[length(out) + 1]] <- data.table(
        rep = i, estimator = paste0("ate:", est), estimand = "rd",
        event = g$event, time = g$time, est = g$est, se = g$se,
        ci_lo = g$ci_lo, ci_hi = g$ci_hi,
        stage2_seconds = el, tier = 2L, source = "riskRegression")
    }

    rbindlist(out, fill = TRUE)
  }, error = function(e) {
    n_fail <<- n_fail + 1L
    if (is.null(first_err)) first_err <<- conditionMessage(e)
    NULL
  })
  if (!is.null(res)) rows[[length(rows) + 1]] <- res
  if (i %% 25 == 0) message("  rep ", i, "  (failed so far: ", n_fail, ")")
}

if (!length(rows)) stop("every replicate failed; first error: ", first_err)
res <- rbindlist(rows, fill = TRUE)
out_name <- if (is.na(SHARD)) "r_estimates.parquet" else
  sprintf("r_estimates_%s.parquet", SHARD)
write_parquet(res, file.path(DIR, out_name))
cat("wrote", nrow(res), "rows from", length(rows), "of", length(data_files),
    "replicates\n")
if (n_fail) cat("(", n_fail, " failed; first error: ", first_err, ")\n", sep = "")
