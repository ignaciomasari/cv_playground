"""Append-only results CSV with fixed schema (LP + FT rows share header; n/a cols = NaN)."""
import os

import pandas as pd

KEY = ["backbone", "init", "mode", "frac", "seed"]
RESULT_COLS = KEY + ["in_chans", "n_train", "top1_val", "top1_test",
                     "lp_wd",                                                        # LP only
                     "steps", "final_loss", "train_time_s", "train_peak_mem_MB"]     # FT only


def read_results(path):
    """DataFrame or None; header mismatch (old/corrupt) -> moved to .bak."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        header = f.readline().strip().split(",")
    if header != RESULT_COLS:
        os.replace(path, path + ".bak")
        print("schema mismatch -> moved to", path + ".bak")
        return None
    return pd.read_csv(path)


def log_row(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame([row]).reindex(columns=RESULT_COLS).to_csv(
        path, mode="a", header=not os.path.isfile(path), index=False)


def done_keys(path):
    df = read_results(path)
    if df is None:
        return set()
    return set(map(tuple, df[KEY].astype({"frac": float, "seed": int}).values.tolist()))
