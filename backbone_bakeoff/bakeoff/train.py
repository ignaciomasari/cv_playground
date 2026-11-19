"""One FT run via Lightning: `python -m bakeoff.train --config configs/default.yaml --set frac=0.1 seed=1`."""
import argparse
import math
import time
import warnings
from functools import partial

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import data as bd
from . import models as bm
from .config import load_config
from .results import log_row

# data = in-RAM int16 tensors, prep on GPU -> workers not needed
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def lr_lambda(step, warm, total):
    """LR multiplier: linear warmup (warm steps) -> cosine to 0 at total."""
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def param_groups(model, wd):
    """AdamW groups: no wd on norm/bias/1-d params, pos_embed, cls_token."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n.endswith((".bias", "pos_embed", "cls_token")) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


class FTModule(L.LightningModule):
    """Full FT; prep + D4 aug on GPU after batch transfer."""

    def __init__(self, model, spec, cfg, total_steps):
        super().__init__()
        self.model, self.spec, self.cfg, self.total_steps = model, spec, cfg, total_steps
        self.crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)
        self.last_loss, self.train_time_s = float("nan"), float("nan")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def on_train_start(self):  # time train loop only (as notebook), not setup/teardown
        self._sync(); self._t0 = time.perf_counter()

    def on_train_end(self):
        self._sync(); self.train_time_s = time.perf_counter() - self._t0

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        self.spec.to(self.device)
        return bm.prep(x, self.spec, self.trainer.training, self.cfg.img_size), y

    def training_step(self, batch, _):
        x, y = batch
        loss = self.crit(self(x), y)
        self.last_loss = loss.detach()
        return loss

    def _eval_start(self):
        self.correct, self.count = 0, 0

    def _eval_step(self, batch):
        x, y = batch
        self.correct += (self(x).argmax(1) == y).sum().item()
        self.count += len(y)

    def _eval_end(self):
        self.log("acc", self.correct / max(1, self.count))

    on_validation_epoch_start = on_test_epoch_start = _eval_start
    on_validation_epoch_end = on_test_epoch_end = _eval_end

    def validation_step(self, batch, _):
        self._eval_step(batch)

    def test_step(self, batch, _):
        self._eval_step(batch)

    def configure_optimizers(self):
        c = self.cfg
        opt = torch.optim.AdamW(param_groups(self.model, c.wd), lr=c.lr)
        warm = max(1, int(c.warmup_frac * self.total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_lambda, warm=warm, total=self.total_steps))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def _loader(x, y, bs, train, num_workers=0):
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=train, drop_last=train and len(y) >= bs,
                      num_workers=num_workers, persistent_workers=num_workers > 0,
                      pin_memory=torch.cuda.is_available())


def run(cfg, data=None):
    """One FT run -> results row (also appended to cfg.results_csv). data: {split: (x int16, y)} or load cache."""
    L.seed_everything(cfg.seed, workers=True, verbose=False)
    torch.backends.cudnn.benchmark = True  # as notebook
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    data = data or bd.load_eurosat(cfg.data_root)
    stats = bd.band_stats(data["train"][0]) if cfg.init == "imagenet_13ch" else (None, None)
    model, spec = bm.build(cfg.backbone, cfg.init, cfg.num_classes, band_mean=stats[0], band_std=stats[1],
                           pretrained=cfg.pretrained)
    idx = torch.from_numpy(bd.subset_idx(data["train"][1], cfg.frac, cfg.seed, cfg.num_classes))
    x, y = data["train"][0][idx], data["train"][1][idx]
    total = max(1, len(y) // cfg.bs) * cfg.max_epochs
    module = FTModule(model, spec, cfg, total)
    trainer = L.Trainer(accelerator=cfg.accelerator, devices=1, precision=cfg.precision, max_epochs=cfg.max_epochs,
                        gradient_clip_val=cfg.clip, logger=False, enable_checkpointing=False,
                        enable_progress_bar=cfg.progress, enable_model_summary=False, num_sanity_val_steps=0)
    cuda = trainer.strategy.root_device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(module, _loader(x, y, cfg.bs, True, cfg.num_workers))
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")

    def top1(split, fn):  # fn: trainer.validate | trainer.test
        return fn(module, _loader(*data[split], 256, False, cfg.num_workers), verbose=False)[0]["acc"]
    row = dict(backbone=cfg.backbone, init=cfg.init, mode="ft", frac=cfg.frac, seed=cfg.seed,
               in_chans=bm.in_chans_of(cfg.init), n_train=len(y),
               top1_val=top1("val", trainer.validate), top1_test=top1("test", trainer.test),
               steps=trainer.global_step, final_loss=float(module.last_loss),
               train_time_s=module.train_time_s, train_peak_mem_MB=peak)
    log_row(cfg.results_csv, row)
    return row


def main(argv=None, data=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="YAML (default: built-in = configs/default.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="config overrides")
    a = p.parse_args(argv)
    row = run(load_config(a.config, a.set), data)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return row


if __name__ == "__main__":
    main()
