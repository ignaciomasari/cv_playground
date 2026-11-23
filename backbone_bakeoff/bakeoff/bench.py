"""Throughput + peak memory: `python -m bakeoff.bench --out bench.csv`."""
import argparse
import gc
import time
from collections.abc import Sequence

import pandas as pd
import timm
import torch
import torch.nn.functional as F

from .models import BACKBONES


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def imgs_per_s(model: torch.nn.Module, in_chans: int = 3, bs: int = 64, img: int = 224, train: bool = False,
               warmup: int = 10, iters: int = 50, dtype: torch.dtype = torch.bfloat16,
               device: torch.device | str = "cuda", num_classes: int = 10) -> tuple[float, float]:
    """img/s + peak MB (NaN off-GPU). Synthetic batch, autocast, channels_last.
    train=False: inference_mode fwd; train=True: fwd+bwd+AdamW step."""
    device = torch.device(device)
    model = model.to(device, memory_format=torch.channels_last)
    x = torch.randn(bs, in_chans, img, img, device=device).contiguous(memory_format=torch.channels_last)
    if train:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        y = torch.randint(0, num_classes, (bs,), device=device)

        def step():
            with torch.autocast(device.type, dtype=dtype):
                loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

        @torch.inference_mode()
        def step():
            with torch.autocast(device.type, dtype=dtype):
                model(x)
    for _ in range(warmup):
        step()
    _sync(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(device)
    throughput = bs * iters / (time.perf_counter() - start_time)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    return throughput, peak_mem_mb


def bench_backbones(backbones: Sequence[str] | None = None, in_chans: Sequence[int] = (3, 13), num_classes: int = 10,
                    **bench_kwargs) -> pd.DataFrame:
    """One row per (backbone, in_chans). Random init: weights don't affect speed."""
    rows = []
    for backbone in backbones or BACKBONES:
        for chans in in_chans:
            model = timm.create_model(BACKBONES[backbone].arch, pretrained=False, in_chans=chans,
                                      num_classes=num_classes)
            infer_ips, infer_mem = imgs_per_s(model, chans, train=False, num_classes=num_classes, **bench_kwargs)
            train_ips, train_mem = imgs_per_s(model, chans, train=True, num_classes=num_classes, **bench_kwargs)
            n_params = sum(param.numel() for param in model.parameters())
            rows.append(dict(backbone=backbone, in_chans=chans, params_M=n_params / 1e6,
                             infer_ips=infer_ips, infer_mem_MB=infer_mem, train_ips=train_ips, train_mem_MB=train_mem))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbones", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--img", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    bench = bench_backbones(args.backbones, bs=args.bs, img=args.img, warmup=args.warmup, iters=args.iters,
                            device=args.device)
    bench.to_csv(args.out, index=False)
    print(bench.round(1).to_string(index=False))
    return bench


if __name__ == "__main__":
    main()
