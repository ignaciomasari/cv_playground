import json

import pytest
import yaml

from bakeoff.analysis import ft_lp_gap, paired_delta, summary, tag_by_grid
from bakeoff.config import config_hash, load_config, recipe
from bakeoff.models import in_chans_of
from bakeoff.results import log_row, read_results


def row(top1, **kv):
    """Results row as train.run writes it (config JSON + hash)."""
    cfg = load_config(None, [f"{k}={v}" for k, v in kv.items()])
    return dict(backbone=cfg.backbone, init=cfg.init, mode=cfg.mode, frac=cfg.frac, seed=cfg.seed,
                in_chans=in_chans_of(cfg.init), top1_test=top1,
                config_hash=config_hash(cfg), config=json.dumps(recipe(cfg), sort_keys=True))


def results(tmp_path, rows):
    p = str(tmp_path / "r.csv")
    for r in rows:
        log_row(p, r)
    return read_results(p)  # through CSV: as the notebook reads it


# --- summary table
def test_summary_splits_by_varied_hparam_not_by_resolved_epochs(tmp_path):
    df = results(tmp_path, [row(0.90, lr=2e-4, seed=0), row(0.92, lr=2e-4, seed=1),
                            row(0.80, lr=1e-3, seed=0), row(0.82, lr=1e-3, seed=1),
                            row(0.70, frac=0.01, lr=2e-4, seed=0)])  # 1% -> 200 epochs vs 50
    s = summary(df)
    assert "lr" in s.columns and "epochs" not in s.columns
    got = s.set_index(["frac", "lr"])
    assert got.loc[(0.1, 2e-4), "top1"] == pytest.approx(0.91) and got.loc[(0.1, 2e-4), "n"] == 2
    assert got.loc[(0.1, 1e-3), "top1"] == pytest.approx(0.81)
    assert len(s) == 3


# --- paired deltas (same frac + seed = same subset)
def test_paired_delta_matches_seed_frac_and_recipe(tmp_path):
    S2, IN = dict(init="s2_ssl"), dict(init="imagenet_rgb")
    df = results(tmp_path, [row(0.95, seed=0, **S2), row(0.90, seed=0, **IN),
                            row(0.93, seed=1, **S2), row(0.91, seed=1, **IN),
                            row(0.99, seed=2, **S2),                            # unpaired -> ignored
                            row(0.50, seed=0, frac=0.01, **IN),                 # other frac
                            row(0.80, seed=0, lr=1e-3, **S2), row(0.85, seed=0, lr=1e-3, **IN)])
    d = paired_delta(df, "init", "s2_ssl", "imagenet_rgb").set_index(["frac", "lr"])
    assert d.loc[(0.1, 2e-4), "mean"] == pytest.approx(3.5) and d.loc[(0.1, 2e-4), "n"] == 2  # (5 + 2) / 2 pp
    assert d.loc[(0.1, 1e-3), "mean"] == pytest.approx(-5.0)
    assert len(d) == 2


def test_ft_lp_gap_pairs_each_ft_recipe_with_lp_of_same_seed(tmp_path):
    df = results(tmp_path, [row(0.80, mode="lp", seed=0), row(0.84, mode="lp", seed=1),
                            row(0.95, seed=0), row(0.96, seed=1),
                            row(0.90, seed=0, drop_path_rate=0.1),
                            row(0.70, mode="lp", seed=0, frac=0.01)])  # no FT partner
    g = ft_lp_gap(df).set_index("drop_path_rate")
    assert g.loc[0.0, "mean"] == pytest.approx(13.5) and g.loc[0.0, "n"] == 2  # (15 + 12) / 2
    assert g.loc[0.1, "mean"] == pytest.approx(10.0) and g.loc[0.1, "n"] == 1
    assert len(g) == 2 and "lr" not in g.columns


def test_explicit_var_pools_tuned_recipes(tmp_path):
    # tuned bake-off: each init its own lr -> auto split never pairs; var=[] pairs on seed only
    df = results(tmp_path, [row(0.95, init="s2_ssl", lr=1e-3), row(0.90, init="imagenet_rgb", lr=5e-4)])
    assert paired_delta(df, "init", "s2_ssl", "imagenet_rgb").empty
    d = paired_delta(df, "init", "s2_ssl", "imagenet_rgb", var=[])
    assert d["mean"].tolist() == pytest.approx([5.0]) and "lr" not in d
    assert paired_delta(df, "lr", 1e-3, 5e-4, var=[]).empty  # different init -> no pair


# --- fixed vs tuned (rows tagged by which grid's hashes they match)
def test_tag_fixed_vs_tuned_side_by_side(tmp_path):
    from test_grid import DEFAULT

    def g(name, **extra):
        (tmp_path / name).write_text(yaml.safe_dump(
            {"base": DEFAULT, "set": {"frac": 0.1}, "grid": {"mode": ["lp", "ft"], "seed": [0, 1]}, **extra}))
        return str(tmp_path / name)
    grids = {"fixed": g("grid.yaml"),
             "tuned": g("tuned.yaml", rules=[{"when": {"backbone": "resnet50"}, "set": {"lr": 1e-3}}])}
    df = results(tmp_path, [row(0.80, mode="lp", seed=0), row(0.82, mode="lp", seed=1),   # in both grids
                            row(0.90, seed=0), row(0.92, seed=1),                          # fixed FT
                            row(0.94, seed=0, lr=1e-3), row(0.95, seed=1, lr=1e-3),        # tuned FT
                            row(0.99, seed=0, lr=5e-4)])                                   # sweep only
    t = tag_by_grid(df, grids)
    s = summary(t, var=["recipe"]).set_index(["mode", "recipe"]).top1
    assert s.to_dict() == pytest.approx({("lp", "fixed"): 0.81, ("lp", "tuned"): 0.81,
                                         ("ft", "fixed"): 0.91, ("ft", "tuned"): 0.945})
    d = paired_delta(t, "recipe", "tuned", "fixed", var=[]).set_index("mode")["mean"]
    assert d.to_dict() == pytest.approx({"ft": 3.5, "lp": 0.0})
