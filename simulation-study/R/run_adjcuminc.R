#!/usr/bin/env Rscript
# Rung 2 of the validation ladder: run Hage et al.'s own estimators on their own
# data-generating code, and record the per-arm cumulative incidence.
#
#   Rscript R/run_adjcuminc.R --scenarios s1,s2,s3,s4 --n 1500 --reps 100 \
#       --times 0.25,0.5,1.0 --out results/validation/adjcuminc_est.parquet
#
# Data come from their `confCSH()`, so nothing here depends on the Python port
# being right -- that is what makes this an independent reference point.
#
# Their four scenarios:
#   s1  both models correctly specified
#   s2  treatment model misspecified  (control arm's covariates resampled, so
#                                      P(A=1|X) stops being logistic)
#   s3  outcome model misspecified    (step in the baseline hazard at x2 > 1)
#   s4  both

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
  i <- match(flag, args)
  if (is.na(i)) default else args[[i + 1]]
}
scenarios <- strsplit(get_arg("--scenarios", "s1,s2,s3,s4"), ",")[[1]]
N <- as.integer(get_arg("--n", "1500"))
REPS <- as.integer(get_arg("--reps", "100"))
TIMES <- as.numeric(strsplit(get_arg("--times", "0.25,0.5,1.0"), ",")[[1]])
SEED <- as.integer(get_arg("--seed", "20250301"))
OUT <- get_arg("--out", "results/validation/adjcuminc_est.parquet")

THETA <- c(-1, -0.5)
BETA <- list(c(1, -1, 0.5), c(-1, 1, -0.5))
SAMPLE_COEF <- c(1, -1, 1)

SCEN <- list(
  s1 = list(control_ref = FALSE, base_haz = c(0, 0)),
  s2 = list(control_ref = TRUE,  base_haz = c(0, 0)),
  s3 = list(control_ref = FALSE, base_haz = c(2, -2)),
  s4 = list(control_ref = TRUE,  base_haz = c(2, -2))
)

# The analyst always fits the same (linear, logistic) models; whether they are
# correct is a property of the scenario, which is their convention.
FML_OUT <- Hist(obs_time, status) ~ treatment + x1 + x2
FML_TRT <- treatment ~ x1 + x2

tidy_curve <- function(df, estimator, rep, scenario, elapsed) {
  dt <- as.data.table(df)
  long <- melt(dt, id.vars = c("time", "strata"), measure.vars = c("1", "2"),
               variable.name = "event", value.name = "risk")
  long[, arm := ifelse(grepl("treat$", strata), 1L, 0L)]
  long[, .(scenario = scenario, rep = rep, estimator = estimator,
           arm, event = as.integer(as.character(event)),
           time = as.numeric(time), risk = as.numeric(risk),
           seconds = elapsed)]
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
        event = cause, time = TIMES, risk = as.numeric(pr[[a]]),
        seconds = elapsed
      )
    }
  }
  rbindlist(out)
}

results <- list()
for (sc in scenarios) {
  cfg <- SCEN[[sc]]
  if (is.null(cfg)) stop("unknown scenario ", sc)
  message("scenario ", sc, " (", REPS, " reps, N = ", N, ")")
  set.seed(SEED + which(names(SCEN) == sc))
  for (r in seq_len(REPS)) {
    df <- confCSH(N = N, k = 2, theta_coef = THETA, beta_coef = BETA,
                  sample_coef = SAMPLE_COEF, control_ref = cfg$control_ref,
                  base_haz = cfg$base_haz)

    t0 <- Sys.time()
    crude <- prodlim(Hist(obs_time, status) ~ treatment, data = df)
    e_crude <- as.numeric(Sys.time() - t0, units = "secs")
    results[[length(results) + 1]] <-
      curve_from_prodlim(crude, "crude", r, sc, e_crude)

    t0 <- Sys.time()
    w <- adjIPW(FML_TRT, data = df)
    ipw_fit <- prodlim(Hist(obs_time, status) ~ treatment, data = df,
                       caseweights = w)
    e_ipw <- as.numeric(Sys.time() - t0, units = "secs")
    results[[length(results) + 1]] <-
      curve_from_prodlim(ipw_fit, "adjIPW", r, sc, e_ipw)

    t0 <- Sys.time()
    or <- adjOR(FML_OUT, strata = "treatment", data = df, times = TIMES)
    e_or <- as.numeric(Sys.time() - t0, units = "secs")
    results[[length(results) + 1]] <- tidy_curve(or, "adjOR", r, sc, e_or)

    t0 <- Sys.time()
    dr <- adjDR(FML_OUT, strata = "treatment", data = df, times = TIMES)
    e_dr <- as.numeric(Sys.time() - t0, units = "secs")
    results[[length(results) + 1]] <- tidy_curve(dr, "adjDR", r, sc, e_dr)

    if (r %% 25 == 0) message("  rep ", r)
  }
}

res <- rbindlist(results)
dir.create(dirname(OUT), recursive = TRUE, showWarnings = FALSE)
write_parquet(res, OUT)
cat("wrote", nrow(res), "rows to", OUT, "\n")
