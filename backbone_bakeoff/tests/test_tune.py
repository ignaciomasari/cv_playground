import pytest
import yaml

from bakeoff.grid import expand
from bakeoff.results import log_row
from bakeoff.tune import pick

from test_grid import DEFAULT


def sweep_file(tmp_path):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump({"base": DEFAULT, "set": {"out_dir": str(tmp_path), "mode": "ft", "frac": 0.1},
                                 "grid": {"backbone": ["resnet50", "vit_s16"], "lr": [1e-4, 1e-3],
                                          "drop_path_rate": [0.0, 0.1]}}))
    return str(sweep_path)


def log(run, val, test):
    cfg = run.cfg
    log_row(cfg.results_csv, dict(backbone=cfg.backbone, init=cfg.init, mode=cfg.mode, frac=cfg.frac,
                                  seed=cfg.seed,
                                  top1_val=val, top1_test=test, config_hash=run.run_id))


# --- best per (backbone, init) on val
def test_pick_best_on_val_per_backbone_ignoring_test_and_foreign_rows(tmp_path):
    sweep = sweep_file(tmp_path)
    runs = {(run.cfg.backbone, run.cfg.lr, run.cfg.drop_path_rate): run for run in expand(sweep)}
    special = {("resnet50", 1e-4, 0.1): (0.95, 0.10),   # best val, awful test -> picked
               ("resnet50", 1e-3, 0.0): (0.90, 0.99),   # best test -> not picked
               ("vit_s16", 1e-3, 0.0): (0.93, 0.90)}
    for key, run in runs.items():
        log(run, *special.get(key, (0.80, 0.80)))
    foreign = expand(sweep, ["seed=1"])[0]              # not a sweep run (other seed)
    log(foreign, 0.99, 0.99)
    assert pick(sweep) == {("resnet50", "imagenet_rgb"): dict(lr=1e-4, drop_path_rate=0.1),
                       ("vit_s16", "imagenet_rgb"): dict(lr=1e-3, drop_path_rate=0.0)}


def test_pick_skips_pairs_without_results(tmp_path):
    sweep = sweep_file(tmp_path)
    run = next(run for run in expand(sweep) if run.cfg.backbone == "vit_s16")
    log(run, 0.9, 0.9)
    assert pick(sweep) == {("vit_s16", "imagenet_rgb"): dict(lr=run.cfg.lr, drop_path_rate=run.cfg.drop_path_rate)}
