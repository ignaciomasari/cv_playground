"""Linear probe: frozen backbone -> cached pooled features -> linear head; wd picked on val. Plain torch."""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as bd
from . import models as bm


def device_of(cfg):
    want_gpu = cfg.accelerator in ("auto", "gpu", "cuda")
    return torch.device("cuda" if want_gpu and torch.cuda.is_available() else "cpu")


def feats_path(cfg):
    """Notebook name for default (224px, pretrained, bf16); suffix otherwise."""
    tag = "" if cfg.img_size == 224 else f"_{cfg.img_size}px"
    tag += "" if cfg.pretrained else "_rand"
    tag += "" if cfg.precision.startswith("bf16") else "_fp32"
    return os.path.join(cfg.out_dir, f"feats_{cfg.backbone}_{cfg.init}{tag}.pt")


@torch.inference_mode()
def extract(cfg, data, device):
    """{split: float feats [N, D]} from frozen eval backbone, no aug; cached in out_dir."""
    path = feats_path(cfg)
    if os.path.isfile(path):
        return torch.load(path)
    m, spec = bm.build_from(cfg, data["train"][0], num_classes=0, device=device)  # 0 -> pooled pre-logits
    m.eval()
    amp = device.type == "cuda" and cfg.precision.startswith("bf16")
    feats = {}
    for split, (x, y) in data.items():
        out = []
        for b in bd.iterate(len(y), bd.EVAL_BS, False):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out.append(m(bm.prep(x[b], spec, False, cfg.img_size)).float().cpu())
        feats[split] = torch.cat(out)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save(feats, path)
    return feats


def train_linear(X, y, wd, cfg):
    torch.manual_seed(cfg.seed)
    head = nn.Linear(X.size(1), cfg.num_classes).to(X.device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lp_lr, weight_decay=wd)
    bs = min(cfg.lp_bs, len(y))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.lp_epochs * math.ceil(len(y) / bs))
    for _ in range(cfg.lp_epochs):
        for b in bd.iterate(len(y), bs, True):
            loss = F.cross_entropy(head(X[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return head


@torch.no_grad()
def _acc(head, X, y):
    return (head(X).argmax(1) == y).float().mean().item()


def run_lp(cfg, data):
    """-> LP row fields (no config cols). Standardise w/ subset stats; best wd on val -> test."""
    dev = device_of(cfg)
    feats = extract(cfg, data, dev)
    idx, _, ytr = bd.train_subset(data, cfg)
    Xtr, ytr = feats["train"][idx].to(dev), ytr.to(dev)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    split = lambda s: ((feats[s].to(dev) - mu) / sd, data[s][1].to(dev))
    Xtr = (Xtr - mu) / sd
    (Xva, yva), (Xte, yte) = split("val"), split("test")
    heads = [(wd, train_linear(Xtr, ytr, wd, cfg)) for wd in cfg.lp_wd_grid]
    val, wd, head = max(((_acc(h, Xva, yva), wd, h) for wd, h in heads), key=lambda t: t[0])
    return dict(n_train=len(idx), top1_val=val, top1_test=_acc(head, Xte, yte), lp_wd=wd)
