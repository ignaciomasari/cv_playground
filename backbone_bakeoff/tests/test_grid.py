import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bakeoff.grid import expand, launch, pending
from bakeoff.results import log_row

DEFAULT = str(Path(__file__).parents[1] / "configs" / "default.yaml")


def grid_file(tmp_path, grid, fixed=None):
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump({"base": DEFAULT, "set": {"out_dir": str(tmp_path), **(fixed or {})}, "grid": grid}))
    return str(path)


# --- expansion
def test_cartesian_product_drops_unavailable_combos(tmp_path):
    runs = expand(grid_file(tmp_path, {"backbone": ["resnet50", "convnext_t"], "init": ["imagenet_rgb", "s2_ssl"],
                                       "seed": [0, 1]}, {"epochs": 2}))
    combos = {(run.cfg.backbone, run.cfg.init, run.cfg.seed) for run in runs}
    assert len(runs) == 6 and ("convnext_t", "s2_ssl", 0) not in combos  # no S2-SSL ConvNeXt
    assert all(run.cfg.epochs == 2 for run in runs)


def test_cli_overrides_win_over_grid_file(tmp_path):
    runs = expand(grid_file(tmp_path, {"seed": [0, 1]}, {"lr": 1e-3}), ["lr=5e-4"])
    assert [run.cfg.lr for run in runs] == [5e-4, 5e-4]


@pytest.mark.parametrize("grid", [{"lrr": [1]}, {"aug": ["d4", "flip"]}])
def test_bad_key_or_value_raises_before_launch(tmp_path, grid, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        launch(grid_file(tmp_path, grid), cmd=[sys.executable, "-c", "open('launched', 'w')"])
    assert not Path("launched").exists()


# --- resume by config hash
def test_runs_with_hash_in_results_are_skipped(tmp_path):
    runs = expand(grid_file(tmp_path, {"seed": [0, 1, 2]}))
    log_row(runs[1].cfg.results_csv, dict(backbone="resnet50", seed=1, config_hash=runs[1].run_id))
    assert [run.cfg.seed for run in pending(runs)] == [0, 2]
    changed = expand(grid_file(tmp_path, {"seed": [0, 1, 2]}, {"lr": 1e-3}))
    assert len(pending(changed)) == 3  # recipe changed -> new hashes


# --- failure isolation (real subprocesses, stand-in runner)
FAKE_RUNNER = """
import sys
from bakeoff.config import config_hash, load_config
from bakeoff.results import log_row
argv = sys.argv[1:]
cfg = load_config(argv[1], argv[3:])
if cfg.seed == 1:
    print("boom"); sys.exit(3)
if cfg.seed == 9:
    import time; time.sleep(30)
log_row(cfg.results_csv, dict(backbone=cfg.backbone, seed=cfg.seed, config_hash=config_hash(cfg)))
"""


def test_failed_run_logged_and_grid_continues(tmp_path):
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_RUNNER)
    grid = grid_file(tmp_path, {"seed": [0, 1, 2]})
    out = launch(grid, cmd=[sys.executable, str(script)])
    assert out == dict(ok=2, failed=1, skipped=0)
    assert sorted(pd.read_csv(tmp_path / "results.csv").seed) == [0, 2]
    fails = pd.read_csv(tmp_path / "failures.csv")
    assert len(fails) == 1 and fails.returncode[0] == 3 and "seed=1" in fails.overrides[0]
    assert "boom" in Path(fails.log[0]).read_text() and fails.error[0] == "boom"
    # relaunch: done runs skipped, failed one retried
    assert launch(grid, cmd=[sys.executable, str(script)]) == dict(ok=0, failed=1, skipped=2)


def test_hung_run_times_out_and_grid_continues(tmp_path):
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_RUNNER)
    out = launch(grid_file(tmp_path, {"seed": [9, 0]}), cmd=[sys.executable, str(script)], timeout_s=3)
    assert out == dict(ok=1, failed=1, skipped=0)
    assert pd.read_csv(tmp_path / "failures.csv").error[0] == "timeout after 3s"


# --- per-(backbone, init) rules
def test_rules_override_only_matching_runs_and_cli_wins(tmp_path):
    grid_path = tmp_path / "g.yaml"
    grid_path.write_text(yaml.safe_dump({"base": DEFAULT, "set": {"out_dir": str(tmp_path)},
                                 "grid": {"backbone": ["resnet50", "vit_s16"], "init": ["imagenet_rgb", "s2_ssl"]},
                                 "rules": [{"when": {"backbone": "vit_s16", "init": "s2_ssl"}, "set": {"lr": 1e-3}},
                                           {"when": {"backbone": "resnet50"}, "set": {"wd": 0.1}}]}))
    settings = {(run.cfg.backbone, run.cfg.init): (run.cfg.lr, run.cfg.wd) for run in expand(str(grid_path))}
    assert settings == {("resnet50", "imagenet_rgb"): (2e-4, 0.1), ("resnet50", "s2_ssl"): (2e-4, 0.1),
                  ("vit_s16", "imagenet_rgb"): (2e-4, 0.05), ("vit_s16", "s2_ssl"): (1e-3, 0.05)}
    assert {run.cfg.lr for run in expand(str(grid_path), ["lr=5e-4"])} == {5e-4}
