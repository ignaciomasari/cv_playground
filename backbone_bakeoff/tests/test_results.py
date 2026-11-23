import os

import pandas as pd

from bakeoff.results import RESULT_COLS, done_keys, log_row, read_results


def test_lp_and_ft_rows_share_one_header(tmp_path):
    results_csv = str(tmp_path / "sub" / "r.csv")
    log_row(results_csv, dict(backbone="resnet50", init="s2_ssl", mode="lp", frac=0.1, seed=0, lp_wd=1e-3, top1_test=0.9))
    log_row(results_csv, dict(backbone="resnet50", init="s2_ssl", mode="ft", frac=0.1, seed=0, steps=10, top1_test=0.95))
    df = read_results(results_csv)
    assert list(df.columns) == RESULT_COLS and len(df) == 2
    assert pd.isna(df.loc[0, "steps"]) and pd.isna(df.loc[1, "lp_wd"])


def test_mismatched_header_moved_to_bak(tmp_path):
    results_csv = tmp_path / "r.csv"
    results_csv.write_text("a,b\n1,2\n")
    assert read_results(str(results_csv)) is None
    assert not results_csv.exists() and os.path.exists(str(results_csv) + ".bak")


def test_done_keys_for_resume(tmp_path):
    results_csv = str(tmp_path / "r.csv")
    assert done_keys(results_csv) == set()
    log_row(results_csv, dict(backbone="vit_s16", init="imagenet_rgb", mode="ft", frac=0.01, seed=2))
    assert done_keys(results_csv) == {("vit_s16", "imagenet_rgb", "ft", 0.01, 2)}


def test_header_pinned_to_existing_results_csv():
    # reorder -> existing results.csv moved to .bak -> every run redone
    assert RESULT_COLS == ["backbone", "init", "mode", "frac", "seed", "in_chans", "n_train", "top1_val", "top1_test",
                           "lp_wd", "steps", "final_loss", "train_time_s", "train_peak_mem_MB", "config_hash", "config"]


def test_unknown_result_col_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="top1"):
        log_row(str(tmp_path / "r.csv"), dict(backbone="resnet50", top1=0.9))


def test_result_row_from_run_round_trips(tmp_path):
    from bakeoff.config import load_config
    from bakeoff.results import LPMetrics, ResultRow
    cfg = load_config(None, ["mode=lp", "init=s2_ssl", f"out_dir={tmp_path}"])
    log_row(cfg.results_csv, ResultRow.from_run(cfg, LPMetrics(n_train=5, top1_val=0.8, top1_test=0.7, lp_wd=1e-3)))
    saved = read_results(cfg.results_csv).iloc[0]
    assert (saved.init, saved["mode"], saved.in_chans, saved.lp_wd) == ("s2_ssl", "lp", 13, 1e-3)
    assert pd.isna(saved.steps)
