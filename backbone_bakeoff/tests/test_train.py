import os

import pandas as pd
import pytest
import timm
import torch

import bakeoff.models as bm
from bakeoff.results import RESULT_COLS, read_results
from bakeoff.train import lr_lambda, main, param_groups


# --- warmup -> cosine (values from notebook formula, worked by hand)
@pytest.mark.parametrize("step,expected", [
    (0, 0.1),        # (0+1)/10
    (9, 1.0),        # last warmup step
    (10, 1.0),       # cosine start
    (55, 0.5),       # midpoint of 90 cosine steps
    (100, 0.0),      # end
])
def test_lr_warmup_then_cosine(step, expected):
    assert lr_lambda(step, warm=10, total=100) == pytest.approx(expected, abs=1e-9)


# --- weight decay groups


def test_no_weight_decay_on_norm_bias_and_vit_tokens():
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=3, img_size=32)
    decay, no_decay = param_groups(m, wd=0.05)
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    names = {id(p): n for n, p in m.named_parameters()}
    nd = {names[id(p)] for p in no_decay["params"]}
    d = {names[id(p)] for p in decay["params"]}
    assert {"pos_embed", "cls_token", "norm.weight", "head.bias"} <= nd
    assert {"patch_embed.proj.weight", "head.weight"} <= d
    assert len(nd) + len(d) == len(names)


# --- end-to-end: CLI -> Lightning fit -> one results row


def fake_eurosat(n=24, hw=16):
    g = torch.Generator().manual_seed(0)
    mk = lambda: (torch.randint(0, 4000, (n, 13, hw, hw), generator=g, dtype=torch.int16),
                  torch.arange(n) % 10)
    return {"train": mk(), "val": mk(), "test": mk()}


def test_cli_run_appends_one_ft_row(tmp_path):
    out = str(tmp_path / "results.csv")
    row = main(["--set", "backbone=resnet50", "init=s2_ssl", "frac=1.0", "seed=1", "epochs=2", "bs=8",
                "img_size=32", "pretrained=false", "accelerator=cpu", "precision=32-true",
                "num_workers=0", "progress=false", f"out_dir={tmp_path}"], data=fake_eurosat())
    df = read_results(out)
    assert list(df.columns) == RESULT_COLS and len(df) == 1
    r = df.iloc[0]
    assert (r.backbone, r.init, r["mode"], r.frac, r.seed, r.in_chans) == ("resnet50", "s2_ssl", "ft", 1.0, 1, 13)
    assert (r.n_train, r.steps) == (24, 6)  # 24 // 8 * 2 epochs
    assert 0 <= r.top1_val <= 1 and 0 <= r.top1_test <= 1 and r.final_loss > 0 and r.train_time_s > 0
    assert row["top1_test"] == r.top1_test


# --- linear probe via same CLI


def lp_argv(tmp_path):
    return ["--set", "mode=lp", "backbone=resnet50", "init=s2_ssl", "frac=1.0", "seed=0", "img_size=32",
            "pretrained=false", "accelerator=cpu", "precision=32-true", "lp_epochs=3", "lp_wd_grid=[0.0, 0.01]",
            "progress=false", f"out_dir={tmp_path}"]


def test_lp_cli_writes_lp_row_and_caches_features(tmp_path, monkeypatch):
    row = main(lp_argv(tmp_path), data=fake_eurosat())
    df = read_results(str(tmp_path / "results.csv"))
    r = df.iloc[0]
    assert (r["mode"], r.n_train, r.in_chans) == ("lp", 24, 13)
    assert r.lp_wd in (0.0, 0.01) and 0 <= r.top1_val <= 1 and 0 <= r.top1_test <= 1
    assert pd.isna(r.steps) and r.config_hash == row["config_hash"] and '"lp_epochs": 3' in r.config
    assert any(f.startswith("feats_resnet50_s2_ssl") for f in os.listdir(tmp_path))

    def no_build(*a, **k):
        raise AssertionError("features should come from cache")
    monkeypatch.setattr(bm, "build", no_build)
    main(lp_argv(tmp_path) + ["seed=1"], data=fake_eurosat())
    assert len(read_results(str(tmp_path / "results.csv"))) == 2
