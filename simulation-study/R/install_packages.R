#!/usr/bin/env Rscript
# Verify (and if necessary install) the R comparator stack.
#
#   Rscript R/install_packages.R
#
# concrete is pinned: the tier-1 nuisance-injection path in run_concrete.R uses
# concrete::: internals whose structure is checked at runtime, and a different
# version could change it.

CONCRETE_PINNED <- "1.0.8"

need <- c(
  survival = "survival",
  riskRegression = "riskRegression",
  prodlim = "prodlim",
  data.table = "data.table",
  arrow = "arrow",
  jsonlite = "jsonlite",
  SuperLearner = "SuperLearner",
  nnls = "nnls",
  concrete = "concrete",
  AdjCuminc = "AdjCuminc",
  adjustedCurves = "adjustedCurves"   # optional extra comparator
)

github <- c(
  concrete = "imbroglio-dc/concrete",
  AdjCuminc = "survival-lumc/AdjCuminc"
)

status <- data.frame(package = character(), version = character(),
                     ok = logical(), stringsAsFactors = FALSE)

for (pkg in need) {
  have <- requireNamespace(pkg, quietly = TRUE)
  if (!have && pkg %in% names(github)) {
    message("installing ", pkg, " from ", github[[pkg]])
    try(remotes::install_github(github[[pkg]], upgrade = "never", quiet = TRUE),
        silent = TRUE)
    have <- requireNamespace(pkg, quietly = TRUE)
  }
  ver <- if (have) as.character(packageVersion(pkg)) else NA_character_
  status <- rbind(status, data.frame(package = pkg, version = ver, ok = have))
}

print(status, row.names = FALSE)

cv <- status$version[status$package == "concrete"]
if (!is.na(cv) && cv != CONCRETE_PINNED) {
  warning("concrete ", cv, " != pinned ", CONCRETE_PINNED,
          "; the nuisance-injection path asserts its internal structure and may ",
          "need updating.", call. = FALSE)
}

essential <- c("survival", "riskRegression", "prodlim", "data.table", "arrow",
               "concrete", "AdjCuminc")
missing <- setdiff(essential, status$package[status$ok])
if (length(missing)) {
  stop("missing essential packages: ", paste(missing, collapse = ", "))
}
cat("\nAll essential comparator packages available.\n")
