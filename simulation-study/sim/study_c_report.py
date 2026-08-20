"""Study C reporting: agreement between implementations, then performance.

Two questions, deliberately answered in this order.

**Does the harness agree with itself?** Every implementation gets the same
nuisances, so a disagreement is an implementation difference and nothing else.
The tolerance depends on the tier (see `sim/study_c.py`), and for `concrete` the
gate is the *targeting increment* rather than the CIF level: the two build
different time grids, which shifts the plug-in by ~1e-3 in both `tmle` and
`gcomp` alike and has nothing to do with the update step (FINDINGS 8).

**Then, how do the estimators perform?** Bias, coverage and RMSE against the
closed-form truth, on the risk-difference scale, for every estimator that targets
it.

The conventional cause-specific Cox is scored **separately and on its own
scale**. A cause-specific log hazard ratio is a conditional, non-collapsible
parameter; it is not an estimate of the marginal risk difference and comparing it
to the RD truth would be a category error, so it never enters the RD tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import get_config
from .metrics import summarise, truth_long
from .study_c import CONFIG, collect_study_c
from .truth import cached_truth

__all__ = ["agreement", "performance", "loghr_panel", "build"]

#: Reference implementation every other row is compared against. PyTMLE is the
#: package under study, so it is the baseline by definition, not by merit.
REFERENCE = "tmle"

#: Which comparison each implementation is entitled to, at what tolerance, and
#: -- since one row is now expected to *disagree* -- what agreement would mean.
#: Stated as data rather than prose so the gate cannot drift from the write-up.
#:
#: `expect="diverge"` is not a way of excusing a failure. Before the
#: targeted-update fix (FINDINGS 9) PyTMLE and concrete agreed on the targeting
#: increment to 0.00018, which is what justified the 2e-3 tolerance. PyTMLE has
#: since been fixed and concrete 1.0.8 has not, so the two now differ by roughly
#: the size of the defect, and *agreement* would be the result worth
#: investigating. The check still runs; only its expected direction changed.
TIERS: Dict[str, Dict] = {
    "tmle (concrete)": dict(tier=1, tol=2e-3, against="tmle", increment=True,
                            expect="diverge",
                            note="concrete 1.0.8 still carries the FINDINGS 9 "
                                 "defect; divergence here is the expected result "
                                 "and its size estimates the defect"),
    # The plug-in involves no targeting, so it is unaffected by the defect and
    # must still agree. It is the control that shows the divergence above is the
    # update step rather than the bridge.
    "gcomp (concrete)": dict(tier=1, tol=5e-3, against="gcomp", increment=False,
                             expect="agree",
                             note="plug-in on concrete's coarser grid; unaffected "
                                  "by the targeted-update defect, so this must agree"),
    "ate:GFORMULA": dict(expect="agree", tier=2, tol=1e-2, against="gcomp", increment=False,
                         note="exact Aalen-Johansen vs discrete plug-in"),
    "ate:IPTW": dict(expect="agree", tier=2, tol=1e-2, against="ipw", increment=False, note=""),
    "ate:AIPTW": dict(expect="agree", tier=2, tol=1e-2, against="aipw", increment=False, note=""),
    "adjDR": dict(expect="agree", tier=3, tol=np.inf, against="aipw", increment=False,
                  note="refits internally; qualitative only"),
}


def _rd(d: pd.DataFrame, event: Optional[int] = None) -> pd.DataFrame:
    out = d[(d["estimand"] == "rd") & d["est"].notna()]
    return out if event is None else out[out["event"] == event]


def agreement(d: pd.DataFrame, event: Optional[int] = None) -> pd.DataFrame:
    """Paired per-replicate agreement of every implementation with its counterpart.

    Paired, because all implementations run on the same replicate: the
    replicate-to-replicate variation cancels and what is left is the
    implementation difference. Comparing marginal means instead would hide a
    disagreement that happens to average out.
    """
    w = _rd(d, event).pivot_table(index=["n", "rep", "event", "time"],
                                  columns="estimator", values="est")
    rows: List[Dict] = []
    for name, cfg in TIERS.items():
        base = cfg["against"]
        if name not in w.columns or base not in w.columns:
            continue
        if cfg["increment"]:
            # the update step's own contribution, free of the grid offset that
            # shifts tmle and gcomp together
            gc, gcc = "gcomp", "gcomp (concrete)"
            if gc not in w.columns or gcc not in w.columns:
                continue
            diff = (w[base] - w[gc]) - (w[name] - w[gcc])
            what = "targeting increment"
        else:
            diff = w[base] - w[name]
            what = "level"
        diff = diff.dropna()
        if diff.empty:
            continue
        for n_val, g in diff.groupby(level="n"):
            a = g.to_numpy(dtype=float)
            rows.append({
                "n": int(n_val), "implementation": name, "vs": base,
                "compared": what, "tier": cfg["tier"], "pairs": len(a),
                "mean_diff": float(a.mean()),
                "mean_abs_diff": float(np.abs(a).mean()),
                "max_abs_diff": float(np.abs(a).max()),
                "tolerance": cfg["tol"],
                "expect": cfg.get("expect", "agree"),
                # "as expected", not "agrees": one row is supposed to disagree,
                # and reporting that as a failure would bury a result.
                "as_expected": bool(
                    (np.abs(a).mean() <= cfg["tol"])
                    == (cfg.get("expect", "agree") == "agree")),
                "note": cfg["note"],
            })
    return pd.DataFrame(rows)


def performance(d: pd.DataFrame, config: str = CONFIG,
                cache_dir: Path | str = "results/study_c/_truth",
                n_mc: int = 4_000_000) -> pd.DataFrame:
    """Bias / coverage / RMSE against the closed-form truth, per (n, estimator)."""
    p = get_config(config)
    out: List[pd.DataFrame] = []
    for n_val, g in _rd(d).groupby("n"):
        taus = sorted(g["time"].dropna().unique())
        tr = cached_truth(p, taus, Path(cache_dir), n_mc=n_mc)
        g = g.assign(cell=f"n{int(n_val)}", group=np.nan)
        if "converged" in g:
            g = g.assign(converged=g["converged"].astype("boolean"))
        s = summarise(g, truth_long(tr))
        s["n"] = int(n_val)
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def loghr_panel(d: pd.DataFrame, config: str = CONFIG) -> pd.DataFrame:
    """The conventional analysis, scored against the truth it actually estimates.

    The DGP's ``theta_j`` *is* the true cause-specific log hazard ratio, so the
    conventional Cox can be scored exactly -- just not on the risk-difference
    scale. The second half of this panel is the point: the same data give a
    treatment "effect" of a different sign or magnitude depending on which
    parameter is reported, which is the motivation for a marginal estimator.
    """
    p = get_config(config)
    h = d[(d["estimand"] == "loghr") & d["est"].notna()]
    if h.empty:
        return pd.DataFrame()
    rows = []
    for (n_val, ev), g in h.groupby(["n", "event"]):
        truth = float(p.theta[int(ev) - 1])
        e = g["est"].to_numpy(dtype=float) - truth
        cov = ((g["ci_lo"] <= truth) & (g["ci_hi"] >= truth)).mean()
        rows.append({
            "n": int(n_val), "event": int(ev), "reps": len(g),
            "true_loghr": truth, "mean_loghr": float(g["est"].mean()),
            "bias": float(e.mean()),
            "mc_se": float(np.std(e, ddof=1) / np.sqrt(len(e))) if len(e) > 1 else np.nan,
            "coverage": float(cov),
        })
    return pd.DataFrame(rows)


def build(dirs: Sequence[Path | str], out_dir: Path | str = "results/study_c",
          event: int = 1, config: str = CONFIG) -> Dict[str, pd.DataFrame]:
    """Collect every n-directory, then write the three tables."""
    d = pd.concat([collect_study_c(x) for x in dirs], ignore_index=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out_dir / "study_c_estimates.parquet", index=False)

    tabs = {
        "agreement": agreement(d, event=event),
        "performance": performance(d, config=config,
                                   cache_dir=out_dir / "_truth"),
        "loghr": loghr_panel(d, config=config),
    }
    for name, t in tabs.items():
        if len(t):
            t.to_csv(out_dir / f"study_c_{name}.csv", index=False)
            (out_dir / f"study_c_{name}.md").write_text(t.to_markdown(index=False))

    # the marginal truth, for the panel that sets the hazard ratio beside it
    from .plots_study_c import make_all_c
    from .runner import target_times_for
    from .truth import closed_form

    p = get_config(config)
    taus = target_times_for(p)
    tr = closed_form(p, taus, n_mc=2_000_000).drop_duplicates(["event", "time"])
    tau = max(taus)
    rd_truth = {int(r.event): float(r.rd)
                for _, r in tr[np.isclose(tr["time"], tau)].iterrows()}
    make_all_c(tabs, out_dir / "figures", rd_truth=rd_truth, event=event)
    return tabs


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.study_c_report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="one directory per sample size")
    ap.add_argument("--out-dir", default="results/study_c")
    ap.add_argument("--event", type=int, default=1)
    a = ap.parse_args(argv)
    tabs = build(a.dirs, a.out_dir, event=a.event)
    for name, t in tabs.items():
        print(f"\n### {name}\n")
        print(t.to_string(index=False) if len(t) else "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
