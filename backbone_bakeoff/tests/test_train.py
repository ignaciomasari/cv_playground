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
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=3, img_size=32)
    decay, no_decay = param_groups(model, wd=0.05)
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    names = {id(param): name for name, param in model.named_parameters()}
    no_decay_names = {names[id(param)] for param in no_decay["params"]}
    decay_names = {names[id(param)] for param in decay["params"]}
    assert {"pos_embed", "cls_token", "norm.weight", "head.bias"} <= no_decay_names
    assert {"patch_embed.proj.weight", "head.weight"} <= decay_names
    assert len(no_decay_names) + len(decay_names) == len(names)


# --- end-to-end: CLI -> Lightning fit -> one results row


def fake_eurosat(n=24, hw=16):
    generator = torch.Generator().manual_seed(0)
    make_split = lambda: (torch.randint(0, 4000, (n, 13, hw, hw), generator=generator, dtype=torch.int16),
                          torch.arange(n) % 10)
    return {"train": make_split(), "val": make_split(), "test": make_split()}


def test_cli_run_appends_one_ft_row(tmp_path):
    out = str(tmp_path / "results.csv")
    row = main(["--set", "backbone=resnet50", "init=s2_ssl", "frac=1.0", "seed=1", "epochs=2", "bs=8",
                "img_size=32", "pretrained=false", "accelerator=cpu", "precision=32-true",
                "num_workers=0", "progress=false", f"out_dir={tmp_path}"], data=fake_eurosat())
    df = read_results(out)
    assert list(df.columns) == RESULT_COLS and len(df) == 1
    saved = df.iloc[0]
    assert (saved.backbone, saved.init, saved["mode"], saved.frac, saved.seed, saved.in_chans) == ("resnet50", "s2_ssl", "ft", 1.0, 1, 13)
    assert (saved.n_train, saved.steps) == (24, 6)  # 24 // 8 * 2 epochs
    assert 0 <= saved.top1_val <= 1 and 0 <= saved.top1_test <= 1 and saved.final_loss > 0 and saved.train_time_s > 0
    assert row["top1_test"] == saved.top1_test


# --- linear probe via same CLI


def lp_argv(tmp_path):
    return ["--set", "mode=lp", "backbone=resnet50", "init=s2_ssl", "frac=1.0", "seed=0", "img_size=32",
            "pretrained=false", "accelerator=cpu", "precision=32-true", "lp_epochs=3", "lp_wd_grid=[0.0, 0.01]",
            "progress=false", f"out_dir={tmp_path}"]


def test_lp_cli_writes_lp_row_and_caches_features(tmp_path, monkeypatch):
    row = main(lp_argv(tmp_path), data=fake_eurosat())
    df = read_results(str(tmp_path / "results.csv"))
    saved = df.iloc[0]
    assert (saved["mode"], saved.n_train, saved.in_chans) == ("lp", 24, 13)
    assert saved.lp_wd in (0.0, 0.01) and 0 <= saved.top1_val <= 1 and 0 <= saved.top1_test <= 1
    assert pd.isna(saved.steps) and saved.config_hash == row["config_hash"] and '"lp_epochs": 3' in saved.config
    assert any(f.startswith("feats_resnet50_s2_ssl") for f in os.listdir(tmp_path))

    def no_build(*a, **k):
        raise AssertionError("features should come from cache")
    monkeypatch.setattr(bm, "build", no_build)
    main(lp_argv(tmp_path) + ["seed=1"], data=fake_eurosat())
    assert len(read_results(str(tmp_path / "results.csv"))) == 2
