import pytest
import yaml

from bakeoff.grid import expand
from bakeoff.results import log_row
from bakeoff.tune import pick

from test_grid import DEFAULT


def sweep_file(tmp_path):
    p = tmp_path / "sweep.yaml"
    p.write_text(yaml.safe_dump({"base": DEFAULT, "set": {"out_dir": str(tmp_path), "mode": "ft", "frac": 0.1},
                                 "grid": {"backbone": ["resnet50", "vit_s16"], "lr": [1e-4, 1e-3],
                                          "drop_path_rate": [0.0, 0.1]}}))
    return str(p)


def log(run, val, test):
    c = run.cfg
    log_row(c.results_csv, dict(backbone=c.backbone, init=c.init, mode=c.mode, frac=c.frac, seed=c.seed,
                                top1_val=val, top1_test=test, config_hash=run.run_id))


# --- best per (backbone, init) on val
def test_pick_best_on_val_per_backbone_ignoring_test_and_foreign_rows(tmp_path):
    s = sweep_file(tmp_path)
    runs = {(r.cfg.backbone, r.cfg.lr, r.cfg.drop_path_rate): r for r in expand(s)}
    special = {("resnet50", 1e-4, 0.1): (0.95, 0.10),   # best val, awful test -> picked
               ("resnet50", 1e-3, 0.0): (0.90, 0.99),   # best test -> not picked
               ("vit_s16", 1e-3, 0.0): (0.93, 0.90)}
    for k, r in runs.items():
        log(r, *special.get(k, (0.80, 0.80)))
    foreign = expand(s, ["seed=1"])[0]              # not a sweep run (other seed)
    log(foreign, 0.99, 0.99)
    assert pick(s) == {("resnet50", "imagenet_rgb"): dict(lr=1e-4, drop_path_rate=0.1),
                       ("vit_s16", "imagenet_rgb"): dict(lr=1e-3, drop_path_rate=0.0)}


def test_pick_skips_pairs_without_results(tmp_path):
    s = sweep_file(tmp_path)
    r = next(r for r in expand(s) if r.cfg.backbone == "vit_s16")
    log(r, 0.9, 0.9)
    assert pick(s) == {("vit_s16", "imagenet_rgb"): dict(lr=r.cfg.lr, drop_path_rate=r.cfg.drop_path_rate)}
