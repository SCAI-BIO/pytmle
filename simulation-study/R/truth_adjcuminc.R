#!/usr/bin/env Rscript
# Rung 1 of the validation ladder: export AdjCuminc's own `trueCSH()` so the
# Python closed form can be checked against it.
#
#   Rscript R/truth_adjcuminc.R --out results/validation/truth_r.parquet
#
# `trueCSH()` integrates the exact conditional CIF against the *unconditional*
# covariate distribution (dnorm over the continuous covariate, uniform over the
# three categorical levels). That is the right target for the scenarios where the
# realised covariate law is marginal -- Scenarios 1 and 3 -- but under the
# control-resampling device of Scenarios 2 and 4 the realised law is a mixture,
# so the two definitions come apart. The Python side computes both; this script
# supplies the reference values.

suppressMessages({
  library(AdjCuminc)
  library(data.table)
  library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) default else args[[i + 1]]
}
out_path <- get_arg("--out", "results/validation/truth_r.parquet")
times <- as.numeric(strsplit(get_arg("--times", "0.25,0.5,1.0"), ",")[[1]])

# Hage et al. parameters. lambda_{k,0} = 0 for Scenarios 1-2; = 2 together with
# the -4 * 1{x3 < 1} term for Scenarios 3-4.
THETA <- c(-1, -0.5)
BETA <- list(c(1, -1, 0.5), c(-1, 1, -0.5))

scenarios <- list(
  s1 = list(base_haz = c(0, 0)),
  s2 = list(base_haz = c(0, 0)),
  s3 = list(base_haz = c(2, -2)),
  s4 = list(base_haz = c(2, -2))
)

rows <- list()
for (nm in names(scenarios)) {
  bh <- scenarios[[nm]]$base_haz
  tc <- trueCSH(time = times, k = 2, theta_coef = THETA,
                beta_coef = BETA, base_haz = bh)
  dt <- as.data.table(tc)
  long <- melt(dt, id.vars = c("time", "treatment"),
               measure.vars = c("1", "2"),
               variable.name = "event", value.name = "risk")
  long[, scenario := nm]
  long[, arm := ifelse(treatment == "treat", 1L, 0L)]
  long[, event := as.integer(as.character(event))]
  rows[[nm]] <- long[, .(scenario, arm, event, time, risk)]
}

res <- rbindlist(rows)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(res, out_path)
cat("wrote", nrow(res), "rows to", out_path, "\n")
print(head(res, 8))
