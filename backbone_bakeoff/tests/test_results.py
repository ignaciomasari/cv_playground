import os

import pandas as pd

from bakeoff.results import RESULT_COLS, done_keys, log_row, read_results


def test_lp_and_ft_rows_share_one_header(tmp_path):
    p = str(tmp_path / "sub" / "r.csv")
    log_row(p, dict(backbone="resnet50", init="s2_ssl", mode="lp", frac=0.1, seed=0, lp_wd=1e-3, top1_test=0.9))
    log_row(p, dict(backbone="resnet50", init="s2_ssl", mode="ft", frac=0.1, seed=0, steps=10, top1_test=0.95))
    df = read_results(p)
    assert list(df.columns) == RESULT_COLS and len(df) == 2
    assert pd.isna(df.loc[0, "steps"]) and pd.isna(df.loc[1, "lp_wd"])


def test_mismatched_header_moved_to_bak(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("a,b\n1,2\n")
    assert read_results(str(p)) is None
    assert not p.exists() and os.path.exists(str(p) + ".bak")


def test_done_keys_for_resume(tmp_path):
    p = str(tmp_path / "r.csv")
    assert done_keys(p) == set()
    log_row(p, dict(backbone="vit_s16", init="imagenet_rgb", mode="ft", frac=0.01, seed=2))
    assert done_keys(p) == {("vit_s16", "imagenet_rgb", "ft", 0.01, 2)}
