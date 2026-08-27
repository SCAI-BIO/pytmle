#!/usr/bin/env Rscript
# Run concrete's *second stage only*, on nuisances injected from Python.
#
#   Rscript R/run_concrete_injected.R --dir results/c5diag --reps 30
#
# PyTMLE's targeted update is a port of concrete's, so running both on
# byte-identical initial estimates separates "the algorithm does this" from "the
# port does this". Nothing here re-fits a nuisance: getInitialEstimate is called
# once only to obtain concrete's internal scaffolding (its time grid and the
# shape of the Estimates object), and every component is then overwritten.
#
# concrete's Estimates object, per intervention arm:
#   PropScore       numeric(n)
#   Hazards         list over causes, each n_times x n
#   EvntFreeSurv    n_times x n
#   NuisanceWeight  n_times x n   = 1 / max(pi * G(t-), MinNuisance)
# plus a top-level `Times` attribute. Note the orientation is times x subjects,
# transposed relative to PyTMLE.

suppressMessages({
  library(concrete); library(data.table); library(survival); library(arrow)
})

# data.table defaults to half the cores (10 of 20 on this machine) and concrete
# leans on it heavily, so an unpinned run is silently multi-threaded and its
# runtime is not comparable with anything. The BLAS R links is the *pthreads*
# OpenBLAS build and reads its thread count at load time, so that half has to be
# set by the caller's environment (OPENBLAS_NUM_THREADS/OMP_NUM_THREADS) before
# this process starts -- it cannot be fixed from in here.
setDTthreads(1L)

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[[i + 1]] }
DIR  <- getarg("--dir", "results/c5diag")
REPS <- as.integer(getarg("--reps", "30"))
MINN <- as.numeric(getarg("--min-nuisance", "0.01"))
# Replicate range, so several processes can share one directory. R is
# single-threaded here and the per-replicate cost is dominated by the
# getInitialEstimate scaffolding, so without this a large cell is a wall-clock
# day. Defaults reproduce the previous whole-range behaviour exactly.
FROM  <- suppressWarnings(as.integer(getarg("--from", NA)))
TO    <- suppressWarnings(as.integer(getarg("--to", NA)))
SHARD <- getarg("--shard", NA)
# Timing repeats for the matched benchmark. The fastest run is kept: a slow one
# means something else touched the machine, a fast one cannot mean less work was
# done. 1 reproduces the previous single-pass behaviour exactly.
REPEATS <- as.integer(getarg("--repeats", "1"))

taus <- as.numeric(read_parquet(file.path(DIR, "taus.parquet"))$time)

# Step-function lookup, evaluating our cumulative hazards at concrete's own time
# points. `H` is a step function so reading it at any t is exact, and taking
# `diff` afterwards aggregates our increments correctly over concrete's intervals.
#
# But concrete's `Times` is NOT a subset of our grid, contrary to what an earlier
# version of this comment claimed: concrete builds its own grid (209 points where
# we have 300 at n = 300) and forces the target times onto it, while our grid is
# the unique observed event times and generally contains none of them. So the two
# implementations evaluate the same plug-in on *different discretisations*, which
# is worth ~0.0012 mean / 0.0033 max in the CIF level -- and it moves `gcomp` and
# `tmle` together, since it is not a property of the targeting step. The clean
# comparison of the *algorithms* is the targeting increment `tmle - gcomp`, which
# agrees ~6x more tightly. See FINDINGS 8.
step_at <- function(mat, from_grid, to_grid) {
  idx <- findInterval(to_grid, from_grid)
  idx[idx < 1] <- 1
  mat[, idx, drop = FALSE]
}

# One failing replicate must not cost the whole cell: concrete raises from inside
# an lapply when a fit degenerates, and an unguarded loop loses every replicate
# computed so far. Failures are recorded and reported instead.
out <- list()
eic_out <- list(); n_fail <- 0L; first_err <- NULL

idx <- seq_len(REPS) - 1L
if (!is.na(FROM)) idx <- idx[idx >= FROM]
if (!is.na(TO))   idx <- idx[idx <  TO]
for (i in idx) {
 res <- tryCatch({
  stub <- file.path(DIR, sprintf("rep%03d", i))
  df   <- as.data.table(read_parquet(paste0(stub, "_data.parquet")))
  grid <- as.numeric(read_parquet(paste0(stub, "_grid.parquet"))$time)
  ps1  <- as.numeric(read_parquet(paste0(stub, "_ps.parquet"))$ps1)
  rd   <- function(nm) as.matrix(read_parquet(paste0(stub, "_", nm, ".parquet")))

  dt <- data.table(time = as.numeric(df$event_time),
                   status = as.integer(df$event_indicator),
                   A = as.integer(df$group),
                   d2 = df$d2, d3 = df$d3, w = df$w_cont)

  mod <- list("A" = c("SL.glm"),
              "0" = list(m = Surv(time, status == 0) ~ A + d2 + d3 + w),
              "1" = list(m = Surv(time, status == 1) ~ A + d2 + d3 + w),
              "2" = list(m = Surv(time, status == 2) ~ A + d2 + d3 + w))
  a <- formatArguments(DataTable = dt, EventTime = "time", EventType = "status",
                       Treatment = "A", Intervention = 0:1, TargetTime = taus,
                       TargetEvent = c(1, 2), Model = mod, CVArg = list(V = 2),
                       Verbose = FALSE, GComp = TRUE, MaxUpdateIter = 200,
                       MinNuisance = MINN)
  AL <- lapply(ls(a), function(x) a[[x]]); names(AL) <- ls(a)

  est <- do.call(concrete:::getInitialEstimate,
                 list(Data = AL$DataTable, Model = AL$Model, CVFolds = AL$CVFolds,
                      MinNuisance = AL$MinNuisance, TargetEvent = AL$TargetEvent,
                      TargetTime = AL$TargetTime, Regime = AL$Regime,
                      ReturnModels = FALSE))
  Times <- attr(est, "Times"); nT <- length(Times); n <- nrow(dt)

  # structural assertions: concrete::: internals are not a stable API
  stopifnot(length(est) == 2L,
            all(c("PropScore", "Hazards", "EvntFreeSurv", "NuisanceWeight") %in% names(est[[1]])),
            identical(dim(est[[1]]$EvntFreeSurv), c(nT, n)))

  for (k in seq_along(est)) {
    tag  <- if (names(est)[k] == "A=1") "1" else "0"
    H1   <- t(step_at(rd(paste0("H1_", tag)), grid, Times))   # nT x n
    H2   <- t(step_at(rd(paste0("H2_", tag)), grid, Times))
    HC   <- t(step_at(rd(paste0("HC_", tag)), grid, Times))
    ps   <- if (tag == "1") ps1 else 1 - ps1
    S    <- exp(-(H1 + H2))
    G    <- exp(-HC)
    Glag <- rbind(rep(1, n), G[-nT, , drop = FALSE])

    dn <- dimnames(est[[k]]$Hazards[["1"]])
    h1 <- apply(H1, 2, function(z) diff(c(0, z)))
    h2 <- apply(H2, 2, function(z) diff(c(0, z)))
    dimnames(h1) <- dn; dimnames(h2) <- dn; dimnames(S) <- dn
    attr(h1, "j") <- 1L; attr(h2, "j") <- 2L
    est[[k]]$Hazards[["1"]] <- h1
    est[[k]]$Hazards[["2"]] <- h2
    est[[k]]$EvntFreeSurv <- S
    psv <- ps
    attributes(psv) <- attributes(est[[k]]$PropScore)[c("g.star.intervention", "g.star.obs")]
    est[[k]]$PropScore <- psv
    NW <- 1 / pmax(matrix(ps, nrow = nT, ncol = n, byrow = TRUE) * Glag, MINN)
    dimnames(NW) <- dn
    est[[k]]$NuisanceWeight <- NW
  }
  attr(est, "Times") <- Times

  # --- second stage only: this is what is timed ---------------------------
  # PyTMLE's `stage2_seconds` covers fit() with initial estimates injected, i.e.
  # the influence curve plus the targeted update. getInitialEstimate above is
  # scaffolding whose output is entirely overwritten, so it is excluded here to
  # keep the two numbers like-for-like.
  # The untargeted state, kept aside. doTmleUpdate replaces the hazards with
  # targeted ones, so a second timed repeat over the same object would start
  # from an already-converged fit and measure nothing. R's copy-on-modify makes
  # this snapshot cheap and sufficient: every mutation below reassigns.
  est_clean <- est

  stage2 <- Inf; cpu2 <- NA_real_
  for (.k in seq_len(REPEATS)) {
    est <- est_clean
    t0 <- Sys.time(); p0 <- proc.time()
    est <- concrete:::getEIC(Estimates = est, Data = AL$DataTable, Regime = AL$Regime,
                             TargetEvent = AL$TargetEvent, TargetTime = AL$TargetTime,
                             MinNuisance = MINN, GComp = TRUE)
    SummEIC <- do.call(rbind, lapply(seq_along(est), function(z)
      cbind(Trt = names(est)[z], est[[z]][["SummEIC"]])))
    NormPnEIC <- concrete:::getNormPnEIC(
      SummEIC[Time %in% AL$TargetTime & Event %in% AL$TargetEvent, PnEIC])
    est <- concrete:::doTmleUpdate(Estimates = est, SummEIC = SummEIC, Data = AL$DataTable,
                                   TargetEvent = AL$TargetEvent, TargetTime = AL$TargetTime,
                                   MaxUpdateIter = 200, OneStepEps = AL$OneStepEps,
                                   NormPnEIC = NormPnEIC, Verbose = FALSE)
    .w <- as.numeric(Sys.time() - t0, units = "secs")
    if (.w < stage2) {
      stage2 <- .w
      cpu2 <- sum((proc.time() - p0)[c("user.self", "sys.self")])
    }
  }
  attr(est, "TargetTime") <- AL$TargetTime; attr(est, "T.tilde") <- dt$time
  attr(est, "TargetEvent") <- AL$TargetEvent; attr(est, "Delta") <- dt$status
  attr(est, "GComp") <- TRUE
  class(est) <- union("ConcreteEst", class(est))

  # The *post-update* summary: doTmleUpdate refreshes SummEIC on every accepted
  # step, so after it returns this is Pn D*(Q*). The SummEIC built above, before
  # the update, is a different object and must not be reused here. Columns match
  # PyTMLE's summarize_ic exactly, which is what makes them comparable at all.
  # Extracted after the clock stops so it cannot inflate the runtime.
  eic <- do.call(rbind, lapply(seq_along(est), function(z)
    cbind(Trt = names(est)[z], as.data.table(est[[z]][["SummEIC"]]))))
  eic <- as.data.table(eic)
  norms <- attr(est, "NormPnEICs")
  eic[, `:=`(rep = i, n = n, n_times = length(attr(est, "Times")),
             norm_pn_eic_first = if (length(norms)) norms[[1]] else NA_real_,
             norm_pn_eic_last  = if (length(norms)) norms[[length(norms)]] else NA_real_)]
  # plain `<-`: the tryCatch body evaluates in the calling frame, which here is
  # the global environment. `<<-` on a replacement form (`x[[i]] <<- v`) starts
  # its lookup in the *parent* environment, which at top level skips globalenv
  # and searches the package path -- so it fails with "object not found".
  eic_out[[length(eic_out) + 1L]] <- eic

  # "Risk" as well as "RD": concrete computes a per-arm absolute-risk SE
  # (seEIC/sqrt(n), getOutput.R:170) that PyTMLE also emits, and asking only for
  # the difference threw away half the standard errors available to compare.
  o <- as.data.table(getOutput(est, Estimand = c("RD", "Risk"), GComp = TRUE))
  conv <- attr(est, "TmleConverged")
  o[, `:=`(rep = i, converged = isTRUE(conv$converged), steps = conv$step,
           stage2_seconds = stage2, stage2_cpu_seconds = cpu2,
           n_times = length(attr(est, "Times")), n = n)]
  o
 }, error = function(e) { n_fail <<- n_fail + 1L
                          if (is.null(first_err)) first_err <<- conditionMessage(e)
                          NULL })
 if (!is.null(res)) out[[length(out) + 1]] <- res
 if ((i + 1) %% 25 == 0) message("  rep ", i + 1, "  (failed so far: ", n_fail, ")")
}

if (!length(out)) stop("every replicate failed; first error: ", first_err)
res <- rbindlist(out, fill = TRUE)
out_name <- if (is.na(SHARD)) "concrete_estimates.parquet" else
  sprintf("concrete_estimates_%s.parquet", SHARD)
write_parquet(res, file.path(DIR, out_name))

if (length(eic_out)) {
  eic_res <- rbindlist(eic_out, fill = TRUE)
  setnames(eic_res, c("PnEIC", "seEIC", "seEIC/(sqrt(n)log(n))"),
           c("pn_eic", "se_eic", "eic_crit"), skip_absent = TRUE)
  setnames(eic_res, c("Time", "Event"), c("time", "event"), skip_absent = TRUE)
  # concrete labels arms by the regime string; map to the same 1/0 the Python
  # side uses, and refuse an unrecognised label rather than emitting NaN
  trt <- as.character(eic_res$Trt)
  grp <- ifelse(grepl("1", trt), 1L, ifelse(grepl("0", trt), 0L, NA_integer_))
  if (anyNA(grp)) stop("unrecognised intervention label(s): ",
                       paste(unique(trt[is.na(grp)]), collapse = ", "))
  eic_res[, `:=`(group = grp, estimator = "tmle (concrete)", source = "concrete")]
  eic_res[, Trt := NULL]
  eic_name <- if (is.na(SHARD)) "concrete_eic.parquet" else
    sprintf("concrete_eic_%s.parquet", SHARD)
  write_parquet(eic_res, file.path(DIR, eic_name))
}
cat("wrote", nrow(res), "rows from", length(out), "of", length(idx), "replicates")
if (n_fail) cat("  (", n_fail, " failed; first error: ", first_err, ")", sep = "")
cat("\n")
