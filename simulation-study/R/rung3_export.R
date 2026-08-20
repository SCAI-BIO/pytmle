#!/usr/bin/env Rscript
# Rung 3 of the validation ladder: generate replicates with Hage et al.'s own
# `confCSH()`, run their estimators, and export both the datasets *and* the
# fitted nuisances so the Python estimators can run on byte-identical inputs.
#
#   Rscript R/rung3_export.R --scenarios s1,s2,s3,s4 --n 500 --reps 50 \
#       --times 0.25,0.5,1.0 --out-dir results/validation/rung3
#
# Doing the generation, their estimators and the nuisance fitting in one pass is
# what guarantees alignment: no seed logic is duplicated, so there is no way for
# the two languages to drift onto different data.
#
# Nuisances are exported as full (n x K) counterfactual cumulative hazards rather
# than as model coefficients. Reconstructing from coefficients would depend on
# matching R's and scikit-survival's centering conventions exactly, and a silent
# mismatch there is precisely the kind of thing this rung exists to rule out.

suppressMessages({
  library(AdjCuminc)
  library(prodlim)
  library(survival)
  library(riskRegression)
  library(data.table)
  library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args); if (is.na(i)) default else args[[i + 1]]
}
scenarios <- strsplit(get_arg("--scenarios", "s1,s2,s3,s4"), ",")[[1]]
N <- as.integer(get_arg("--n", "500"))
REPS <- as.integer(get_arg("--reps", "50"))
TIMES <- as.numeric(strsplit(get_arg("--times", "0.25,0.5,1.0"), ",")[[1]])
SEED <- as.integer(get_arg("--seed", "20250301"))
OUT_DIR <- get_arg("--out-dir", "results/validation/rung3")

THETA <- c(-1, -0.5); BETA <- list(c(1, -1, 0.5), c(-1, 1, -0.5))
SAMPLE_COEF <- c(1, -1, 1)
SCEN <- list(
  s1 = list(control_ref = FALSE, base_haz = c(0, 0)),
  s2 = list(control_ref = TRUE,  base_haz = c(0, 0)),
  s3 = list(control_ref = FALSE, base_haz = c(2, -2)),
  s4 = list(control_ref = TRUE,  base_haz = c(2, -2))
)
FML_OUT <- Hist(obs_time, status) ~ treatment + x1 + x2
FML_TRT <- treatment ~ x1 + x2

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

tidy_curve <- function(df, estimator, rep, scenario, elapsed) {
  dt <- as.data.table(df)
  long <- melt(dt, id.vars = c("time", "strata"), measure.vars = c("1", "2"),
               variable.name = "event", value.name = "risk")
  long[, .(scenario = scenario, rep = rep, estimator = estimator,
           arm = ifelse(grepl("treat$", strata), 1L, 0L),
           event = as.integer(as.character(event)),
           time = as.numeric(time), risk = as.numeric(risk), seconds = elapsed)]
}

curve_from_prodlim <- function(fit, estimator, rep, scenario, elapsed) {
  nd <- data.frame(treatment = factor(c("control", "treat"),
                                      levels = levels(fit$X$treatment)))
  out <- list()
  for (cause in 1:2) {
    pr <- predict(fit, times = TIMES, cause = cause, newdata = nd)
    for (a in seq_along(pr)) {
      out[[length(out) + 1]] <- data.table(
        scenario = scenario, rep = rep, estimator = estimator,
        arm = ifelse(grepl("treat$", names(pr)[a]), 1L, 0L),
        event = cause, time = TIMES, risk = as.numeric(pr[[a]]), seconds = elapsed)
    }
  }
  rbindlist(out)
}

est_rows <- list()
for (sc in scenarios) {
  cfg <- SCEN[[sc]]
  message("scenario ", sc)
  set.seed(SEED + which(names(SCEN) == sc))
  for (r in seq_len(REPS)) {
    df <- confCSH(N = N, k = 2, theta_coef = THETA, beta_coef = BETA,
                  sample_coef = SAMPLE_COEF, control_ref = cfg$control_ref,
                  base_haz = cfg$base_haz)
    df$.row <- seq_len(nrow(df))

    # --- their estimators -------------------------------------------------
    t0 <- Sys.time()
    w <- adjIPW(FML_TRT, data = df)
    ipw_fit <- prodlim(Hist(obs_time, status) ~ treatment, data = df, caseweights = w)
    e <- as.numeric(Sys.time() - t0, units = "secs")
    est_rows[[length(est_rows) + 1]] <- curve_from_prodlim(ipw_fit, "adjIPW", r, sc, e)

    t0 <- Sys.time()
    or <- adjOR(FML_OUT, strata = "treatment", data = df, times = TIMES)
    e <- as.numeric(Sys.time() - t0, units = "secs")
    est_rows[[length(est_rows) + 1]] <- tidy_curve(or, "adjOR", r, sc, e)

    t0 <- Sys.time()
    dr <- adjDR(FML_OUT, strata = "treatment", data = df, times = TIMES)
    e <- as.numeric(Sys.time() - t0, units = "secs")
    est_rows[[length(est_rows) + 1]] <- tidy_curve(dr, "adjDR", r, sc, e)

    # --- nuisances on the replicate's own event-time grid ------------------
    grid <- sort(unique(df$obs_time))
    d1 <- transform(df, treatment = factor("treat",   levels = levels(df$treatment)))
    d0 <- transform(df, treatment = factor("control", levels = levels(df$treatment)))

    csc <- CSC(FML_OUT, data = df)
    cens <- coxph(Surv(obs_time, status == "0") ~ treatment + x1 + x2,
                  data = df, x = TRUE, y = TRUE)
    trt <- glm(I(treatment == "treat") ~ x1 + x2, data = df, family = binomial)

    ch <- function(fit, nd) predictCox(fit, newdata = nd, times = grid,
                                       type = "cumhazard")$cumhazard
    nz <- list(
      grid = grid,
      ps1 = as.numeric(predict(trt, type = "response")),
      H1_1 = ch(csc$models[[1]], d1), H1_0 = ch(csc$models[[1]], d0),
      H2_1 = ch(csc$models[[2]], d1), H2_0 = ch(csc$models[[2]], d0),
      HC_1 = ch(cens, d1),            HC_0 = ch(cens, d0)
    )

    stub <- file.path(OUT_DIR, sprintf("%s_rep%03d", sc, r))
    write_parquet(as.data.table(df), paste0(stub, "_data.parquet"))
    write_parquet(data.table(time = grid), paste0(stub, "_grid.parquet"))
    write_parquet(data.table(ps1 = nz$ps1), paste0(stub, "_ps.parquet"))
    for (nm in c("H1_1", "H1_0", "H2_1", "H2_0", "HC_1", "HC_0")) {
      write_parquet(as.data.frame(nz[[nm]]), paste0(stub, "_", nm, ".parquet"))
    }
    if (r %% 10 == 0) message("  rep ", r)
  }
}

write_parquet(rbindlist(est_rows), file.path(OUT_DIR, "their_estimates.parquet"))
cat("wrote replicates and nuisances to", OUT_DIR, "\n")
