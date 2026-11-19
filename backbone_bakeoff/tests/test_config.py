from pathlib import Path

import pytest

from bakeoff.config import config_hash, load_config


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


# --- config hash (resume / grid identity)


def test_hash_ignores_key_order_and_system_fields(tmp_path):
    a = load_config(write(tmp_path, "lr: 1.0e-4\nseed: 1\n"))
    b = load_config(None, ["seed=1", "lr=1.0e-4", "num_workers=4", "progress=false",
                           f"out_dir={tmp_path}", "accelerator=cpu"])
    assert config_hash(a) == config_hash(b)


@pytest.mark.parametrize("change", ["lr=1.0e-3", "drop_path_rate=0.1", "aug=none", "seed=1", "mode=lp"])
def test_hash_changes_with_recipe(change):
    assert config_hash(load_config(None, [change])) != config_hash(load_config())


@pytest.mark.parametrize("bad", ["aug=flip", "mode=kd"])
def test_invalid_choice_raises(bad):
    with pytest.raises(ValueError, match=bad.split("=")[1]):
        load_config(None, [bad])


def test_hash_ignores_fields_of_other_mode():
    lp = lambda *o: config_hash(load_config(None, ["mode=lp", *o]))
    ft = lambda *o: config_hash(load_config(None, ["mode=ft", *o]))
    assert lp("lr=1.0e-3", "drop_path_rate=0.1") == lp()
    assert lp("lp_lr=1.0e-2") != lp()
    assert ft("lp_epochs=5") == ft()


def test_numbers_coerced_to_field_types():
    # YAML reads "1e-3" (no dot) as str
    cfg = load_config(None, ["lr=1e-3", "lp_lr=5e-4", "bs=32", "drop_path_rate=0"])
    assert (cfg.lr, cfg.lp_lr, cfg.bs, cfg.drop_path_rate) == (1e-3, 5e-4, 32, 0.0)
    assert isinstance(cfg.drop_path_rate, float)
    assert config_hash(cfg) == config_hash(load_config(None, ["lr=1.0e-3", "lp_lr=5.0e-4", "bs=32"]))


def test_hash_uses_resolved_epochs():
    base = config_hash(load_config(None, ["frac=0.1"]))
    assert config_hash(load_config(None, ["frac=0.1", "epochs_by_frac={0.01: 1, 0.1: 50, 1.0: 1}"])) == base
    assert config_hash(load_config(None, ["frac=0.1", "epochs=50"])) == base
    assert config_hash(load_config(None, ["frac=0.1", "epochs=51"])) != base
