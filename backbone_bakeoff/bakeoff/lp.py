"""Linear probe: frozen backbone -> cached pooled features -> linear head; wd picked on val. Plain torch."""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as bd
from . import models as bm
from .config import RunConfig
from .data import Data
from .results import LPMetrics


def device_of(cfg: RunConfig) -> torch.device:
    want_gpu = cfg.accelerator in ("auto", "gpu", "cuda")
    return torch.device("cuda" if want_gpu and torch.cuda.is_available() else "cpu")


def feats_path(cfg: RunConfig) -> str:
    """Notebook name for default (224px, pretrained, bf16); suffix otherwise."""
    tag = "" if cfg.img_size == 224 else f"_{cfg.img_size}px"
    tag += "" if cfg.pretrained else "_rand"
    tag += "" if cfg.precision.startswith("bf16") else "_fp32"
    return os.path.join(cfg.out_dir, f"feats_{cfg.backbone}_{cfg.init}{tag}.pt")


@torch.inference_mode()
def extract(cfg: RunConfig, data: Data, device: torch.device) -> dict[str, torch.Tensor]:
    """{split: float feats [N, D]} from frozen eval backbone, no aug; cached in out_dir."""
    path = feats_path(cfg)
    if os.path.isfile(path):
        return torch.load(path)
    model, spec = bm.build_from(cfg, data["train"][0], num_classes=0, device=device)  # 0 -> pooled pre-logits
    model.eval()
    amp = device.type == "cuda" and cfg.precision.startswith("bf16")
    feats = {}
    for split, (x, y) in data.items():
        batch_feats = []
        for batch_idx in bd.iterate(len(y), bd.EVAL_BS, False):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                batch_feats.append(model(bm.prep(x[batch_idx], spec, False, cfg.img_size)).float().cpu())
        feats[split] = torch.cat(batch_feats)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save(feats, path)
    return feats


def train_linear(feats: torch.Tensor, y: torch.Tensor, wd: float, cfg: RunConfig) -> nn.Linear:
    torch.manual_seed(cfg.seed)
    head = nn.Linear(feats.size(1), cfg.num_classes).to(feats.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lp_lr, weight_decay=wd)
    batch_size = min(cfg.lp_bs, len(y))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.lp_epochs * math.ceil(len(y) / batch_size))
    for _ in range(cfg.lp_epochs):
        for batch_idx in bd.iterate(len(y), batch_size, True):
            loss = F.cross_entropy(head(feats[batch_idx]), y[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return head


@torch.no_grad()
def _acc(head: nn.Module, feats: torch.Tensor, y: torch.Tensor) -> float:
    return (head(feats).argmax(1) == y).float().mean().item()


def run_lp(cfg: RunConfig, data: Data) -> LPMetrics:
    """Standardise w/ subset stats; best wd on val -> test."""
    device = device_of(cfg)
    feats = extract(cfg, data, device)
    idx, _, y_train = bd.train_subset(data, cfg)
    train_feats, y_train = feats["train"][idx].to(device), y_train.to(device)
    mu, sd = train_feats.mean(0), train_feats.std(0) + 1e-6
    standardised = lambda split: ((feats[split].to(device) - mu) / sd, data[split][1].to(device))
    train_feats = (train_feats - mu) / sd
    (val_feats, y_val), (test_feats, y_test) = standardised("val"), standardised("test")
    heads = [(wd, train_linear(train_feats, y_train, wd, cfg)) for wd in cfg.lp_wd_grid]
    val_acc, wd, head = max(((_acc(head, val_feats, y_val), wd, head) for wd, head in heads),
                            key=lambda candidate: candidate[0])
    return LPMetrics(n_train=len(idx), top1_val=val_acc, top1_test=_acc(head, test_feats, y_test), lp_wd=wd)
