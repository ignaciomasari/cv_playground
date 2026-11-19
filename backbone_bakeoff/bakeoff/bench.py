"""Throughput + peak memory: `python -m bakeoff.bench --out bench.csv`."""
import argparse
import gc
import time

import pandas as pd
import timm
import torch
import torch.nn.functional as F

from .models import BACKBONES


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def imgs_per_s(model, in_chans=3, bs=64, img=224, train=False, warmup=10, iters=50,
               dtype=torch.bfloat16, device="cuda", num_classes=10):
    """img/s + peak MB (NaN off-GPU). Synthetic batch, autocast, channels_last.
    train=False: inference_mode fwd; train=True: fwd+bwd+AdamW step."""
    dev = torch.device(device)
    model = model.to(dev, memory_format=torch.channels_last)
    x = torch.randn(bs, in_chans, img, img, device=dev).contiguous(memory_format=torch.channels_last)
    if train:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        y = torch.randint(0, num_classes, (bs,), device=dev)

        def step():
            with torch.autocast(dev.type, dtype=dtype):
                loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    else:
        model.eval()

        @torch.inference_mode()
        def step():
            with torch.autocast(dev.type, dtype=dtype):
                model(x)
    for _ in range(warmup):
        step()
    _sync(dev); gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(dev)
    ips = bs * iters / (time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else float("nan")
    return ips, mem


def bench_backbones(backbones=None, in_chans=(3, 13), num_classes=10, **kw):
    """One row per (backbone, in_chans). Random init: weights don't affect speed."""
    rows = []
    for bb in backbones or BACKBONES:
        for ic in in_chans:
            m = timm.create_model(BACKBONES[bb]["arch"], pretrained=False, in_chans=ic, num_classes=num_classes)
            inf_ips, inf_mem = imgs_per_s(m, ic, train=False, num_classes=num_classes, **kw)
            tr_ips, tr_mem = imgs_per_s(m, ic, train=True, num_classes=num_classes, **kw)
            rows.append(dict(backbone=bb, in_chans=ic, params_M=sum(p.numel() for p in m.parameters()) / 1e6,
                             infer_ips=inf_ips, infer_mem_MB=inf_mem, train_ips=tr_ips, train_mem_MB=tr_mem))
            del m; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--backbones", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    df = bench_backbones(a.backbones, bs=a.bs, img=a.img, warmup=a.warmup, iters=a.iters, device=a.device)
    df.to_csv(a.out, index=False)
    print(df.round(1).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
