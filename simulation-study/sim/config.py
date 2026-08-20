"""YAML -> ``Cell`` objects.

A config file has a ``defaults`` block and a list of ``cells``; each cell
inherits the defaults and overrides what it needs. Keeping the design in YAML
rather than in code means a run is reproducible from an artefact that can sit
next to its results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from .nuisance import Spec
from .runner import Cell

__all__ = ["load_config", "load_cells"]

_CELL_FIELDS = {
    "config",
    "n",
    "reps",
    "estimators",
    "min_nuisance",
    "max_updates",
    "tau_quantiles",
    "target_times",
    "concrete_reps",
    "params_override",
}


def load_config(path: Path | str) -> Dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if "cells" not in cfg:
        raise ValueError(f"{path}: config must define a 'cells' list")
    return cfg


def _make_spec(raw) -> Spec:
    if raw is None:
        return Spec()
    if isinstance(raw, str):  # e.g. "correct/correct/wrong"
        parts = raw.split("/")
        if len(parts) != 3:
            raise ValueError(f"spec string must be 'Q/pi/G', got {raw!r}")
        return Spec(*parts)
    return Spec(**raw)


def load_cells(
    path: Path | str,
    only: Optional[Sequence[str]] = None,
    reps: Optional[int] = None,
) -> List[Cell]:
    """Build the cell list, optionally filtered by name and with reps overridden."""
    cfg = load_config(path)
    defaults = dict(cfg.get("defaults", {}))
    default_spec = defaults.pop("spec", None)

    cells: List[Cell] = []
    for raw in cfg["cells"]:
        raw = dict(raw)
        name = raw.pop("name")
        spec = _make_spec(raw.pop("spec", default_spec))
        merged = {**defaults, **raw}
        unknown = set(merged) - _CELL_FIELDS
        if unknown:
            raise ValueError(f"cell {name!r}: unknown keys {sorted(unknown)}")
        if reps is not None:
            merged["reps"] = reps
        cells.append(Cell(name=name, spec=spec, **merged))

    if only:
        wanted = set(only)
        missing = wanted - {c.name for c in cells}
        if missing:
            raise ValueError(f"unknown cells {sorted(missing)}; have {[c.name for c in cells]}")
        cells = [c for c in cells if c.name in wanted]
    return cells
