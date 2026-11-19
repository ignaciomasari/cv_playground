from pathlib import Path

import pytest

from bakeoff.config import load_config


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return str(p)


def test_yaml_values_loaded(tmp_path):
    cfg = load_config(write(tmp_path, "backbone: vit_s16\ninit: s2_ssl\nlr: 1.0e-4\n"))
    assert (cfg.backbone, cfg.init, cfg.lr) == ("vit_s16", "s2_ssl", 1e-4)


def test_overrides_win_and_are_typed(tmp_path):
    cfg = load_config(write(tmp_path, "frac: 0.1\nseed: 0\n"), ["frac=0.01", "seed=2", "pretrained=false"])
    assert cfg.frac == 0.01 and cfg.seed == 2 and cfg.pretrained is False


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError, match="lrr"):
        load_config(write(tmp_path, "lrr: 1\n"))
    with pytest.raises(ValueError, match="sed"):
        load_config(None, ["sed=1"])


def test_epochs_default_from_fraction_table():
    # notebook recipe: 1% -> 200, 10% -> 50, 100% -> 15
    assert [load_config(None, [f"frac={f}"]).max_epochs for f in (0.01, 0.1, 1.0)] == [200, 50, 15]
    assert load_config(None, ["frac=0.1", "epochs=3"]).max_epochs == 3


def test_repo_default_yaml_matches_notebook_recipe():
    cfg = load_config(str(Path(__file__).parents[1] / "configs" / "default.yaml"))
    assert (cfg.lr, cfg.wd, cfg.warmup_frac, cfg.label_smooth, cfg.clip, cfg.bs, cfg.img_size) == \
        (2e-4, 0.05, 0.05, 0.1, 1.0, 64, 224)
