"""Append-only results CSV with fixed schema (LP + FT rows share header; n/a cols = NaN)."""
import dataclasses
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import RunConfig, config_hash, recipe

NAN = math.nan
log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Metrics:
    """Runner output common to LP + FT."""
    n_train: int
    top1_val: float
    top1_test: float


@dataclass(kw_only=True)
class LPMetrics(Metrics):
    lp_wd: float  # best on val


@dataclass(kw_only=True)
class FTMetrics(Metrics):
    steps: int
    final_loss: float
    train_time_s: float
    train_peak_mem_MB: float  # NaN off-GPU


@dataclass(kw_only=True)
class ResultRow:
    """One CSV row. Field order = CSV header: never reorder (other header -> file moved to .bak)."""
    backbone: str
    init: str
    mode: str
    frac: float
    seed: int
    in_chans: int
    n_train: int
    top1_val: float
    top1_test: float
    lp_wd: float = NAN              # LP only
    steps: float = NAN              # FT only (NaN for LP -> float)
    final_loss: float = NAN
    train_time_s: float = NAN
    train_peak_mem_MB: float = NAN
    config_hash: str                # recipe id
    config: str                     # recipe JSON

    @classmethod
    def from_run(cls, cfg: RunConfig, metrics: Metrics) -> "ResultRow":
        return cls(backbone=cfg.backbone, init=str(cfg.init), mode=str(cfg.mode), frac=cfg.frac, seed=cfg.seed,
                   in_chans=cfg.init.in_chans, **dataclasses.asdict(metrics),
                   config_hash=config_hash(cfg), config=json.dumps(recipe(cfg), sort_keys=True))


RESULT_COLS = [result_field.name for result_field in dataclasses.fields(ResultRow)]
KEY = RESULT_COLS[:5]  # backbone, init, mode, frac, seed


def _ensure_schema(path: str) -> bool:
    """Other header (old/corrupt) -> moved to .bak. True if file exists with RESULT_COLS."""
    if not os.path.isfile(path):
        return False
    with open(path) as file:
        header = file.readline().strip().split(",")
    if header != RESULT_COLS:
        os.replace(path, path + ".bak")
        log.warning("schema mismatch -> moved to %s.bak", path)
        return False
    return True


def read_results(path: str) -> pd.DataFrame | None:
    """DataFrame or None (missing / moved to .bak)."""
    return pd.read_csv(path) if _ensure_schema(path) else None


def log_row(path: str, row: ResultRow | Mapping[str, Any]) -> None:
    """Append row (partial mapping ok: missing cols NaN); file with other header moved to .bak first."""
    if isinstance(row, ResultRow):
        row = dataclasses.asdict(row)
    if unknown := set(row) - set(RESULT_COLS):
        raise ValueError(f"unknown result cols: {sorted(unknown)}")
    _ensure_schema(path)
    append_csv(path, pd.DataFrame([row]).reindex(columns=RESULT_COLS))


def append_csv(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.isfile(path), index=False)


def done_keys(path: str) -> set[tuple]:
    df = read_results(path)
    if df is None:
        return set()
    return set(map(tuple, df[KEY].astype({"frac": float, "seed": int}).values.tolist()))


def done_hashes(path: str) -> set[str]:
    """config_hash of finished runs (rows w/o hash, e.g. notebook, ignored)."""
    df = read_results(path)
    return set() if df is None else set(df.config_hash.dropna())
