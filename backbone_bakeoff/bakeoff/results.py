"""Append-only results CSV with fixed schema (LP + FT rows share header; n/a cols = NaN)."""
import os

import pandas as pd

KEY = ["backbone", "init", "mode", "frac", "seed"]
RESULT_COLS = KEY + ["in_chans", "n_train", "top1_val", "top1_test",
                     "lp_wd",                                                        # LP only
                     "steps", "final_loss", "train_time_s", "train_peak_mem_MB",     # FT only
                     "config_hash", "config"]                                        # recipe id + JSON


def _ensure_schema(path):
    """Other header (old/corrupt) -> moved to .bak. True if file exists with RESULT_COLS."""
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        header = f.readline().strip().split(",")
    if header != RESULT_COLS:
        os.replace(path, path + ".bak")
        print("schema mismatch -> moved to", path + ".bak")
        return False
    return True


def read_results(path):
    """DataFrame or None (missing / moved to .bak)."""
    return pd.read_csv(path) if _ensure_schema(path) else None


def log_row(path, row):
    """Append row; file with other header moved to .bak first."""
    _ensure_schema(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame([row]).reindex(columns=RESULT_COLS).to_csv(
        path, mode="a", header=not os.path.isfile(path), index=False)


def done_keys(path):
    df = read_results(path)
    if df is None:
        return set()
    return set(map(tuple, df[KEY].astype({"frac": float, "seed": int}).values.tolist()))
