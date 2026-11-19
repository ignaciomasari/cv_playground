"""One run (FT via Lightning | LP) -> results row: `python -m bakeoff.train --set mode=lp backbone=vit_s16 seed=1`."""
import argparse
import json
import math
import time
import warnings
from functools import partial, partialmethod

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import data as bd
from . import models as bm
from .config import config_hash, load_config, recipe
from .lp import run_lp
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
    for name, param in model.named_parameters():
        skip = param.ndim <= 1 or name.endswith((".bias", "pos_embed", "cls_token"))
        (no_decay if skip else decay).append(param)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


class FTModule(L.LightningModule):
    """Full FT; prep + D4 aug on GPU after batch transfer."""

    def __init__(self, model, spec, cfg, total_steps):
        super().__init__()
        self.model, self.spec, self.cfg, self.total_steps = model, spec, cfg, total_steps
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)
        self.last_loss, self.train_time_s = float("nan"), float("nan")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def on_train_start(self):  # time train loop only (as notebook), not setup/teardown
        self._sync()
        self._start_time = time.perf_counter()

    def on_train_end(self):
        self._sync()
        self.train_time_s = time.perf_counter() - self._start_time

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        self.spec.to(self.device)
        aug = self.trainer.training and self.cfg.aug == "d4"
        return bm.prep(x, self.spec, aug, self.cfg.img_size), y

    def training_step(self, batch, _):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.last_loss = loss.detach()
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def _eval_start(self):
        self.correct, self.count = 0, 0

    def _eval_step(self, batch):
        x, y = batch
        self.correct += (self(x).argmax(1) == y).sum().item()
        self.count += len(y)

    def _eval_end(self, split):
        self.log(f"{split}_acc", self.correct / max(1, self.count))

    on_validation_epoch_start = on_test_epoch_start = _eval_start
    on_validation_epoch_end = partialmethod(_eval_end, "val")
    on_test_epoch_end = partialmethod(_eval_end, "test")

    def validation_step(self, batch, _):
        self._eval_step(batch)

    def test_step(self, batch, _):
        self._eval_step(batch)

    def configure_optimizers(self):
        cfg = self.cfg
        optimizer = torch.optim.AdamW(param_groups(self.model, cfg.wd), lr=cfg.lr)
        warmup_steps = max(1, int(cfg.warmup_frac * self.total_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, partial(lr_lambda, warm=warmup_steps, total=self.total_steps))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def _loader(x, y, batch_size, train, num_workers=0):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=train,
                      drop_last=train and len(y) >= batch_size,
                      num_workers=num_workers, persistent_workers=num_workers > 0,
                      pin_memory=torch.cuda.is_available())


def _loggers(cfg):
    """FT curves (train_loss, lr per step; val acc if val_every) under out_dir/{runs,tb}/<hash>."""
    if not cfg.curves:
        return False
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    run_id = config_hash(cfg)
    loggers = [CSVLogger(cfg.out_dir, name="runs", version=run_id)]
    if cfg.tensorboard:
        loggers.append(TensorBoardLogger(cfg.out_dir, name="tb", version=run_id))
    return loggers


def run_ft(cfg, data):
    """-> FT row fields (no config cols)."""
    torch.backends.cudnn.benchmark = True  # as notebook
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    model, spec = bm.build_from(cfg, data["train"][0])
    _, x, y = bd.train_subset(data, cfg)
    steps_per_epoch = max(1, len(y) // cfg.bs)
    module = FTModule(model, spec, cfg, steps_per_epoch * cfg.max_epochs)
    loggers = _loggers(cfg)
    trainer = L.Trainer(accelerator=cfg.accelerator, devices=1, precision=cfg.precision, max_epochs=cfg.max_epochs,
                        gradient_clip_val=cfg.clip, logger=loggers, enable_checkpointing=False,
                        callbacks=[L.pytorch.callbacks.LearningRateMonitor("step")] if loggers else None,
                        log_every_n_steps=min(10, steps_per_epoch), check_val_every_n_epoch=max(1, cfg.val_every),
                        enable_progress_bar=cfg.progress, enable_model_summary=False, num_sanity_val_steps=0)
    on_cuda = trainer.strategy.root_device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    val_loader = _loader(*data["val"], bd.EVAL_BS, False, cfg.num_workers) if cfg.val_every > 0 else None
    trainer.fit(module, _loader(x, y, cfg.bs, True, cfg.num_workers), val_loader)
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if on_cuda else float("nan")

    def top1(split, evaluate):  # evaluate: trainer.validate | trainer.test
        loader = _loader(*data[split], bd.EVAL_BS, False, cfg.num_workers)
        return evaluate(module, loader, verbose=False)[0][f"{split}_acc"]
    return dict(n_train=len(y), top1_val=top1("val", trainer.validate), top1_test=top1("test", trainer.test),
                steps=trainer.global_step, final_loss=float(module.last_loss),
                train_time_s=module.train_time_s, train_peak_mem_MB=peak_mem_mb)


RUNNERS = {"ft": run_ft, "lp": run_lp}


def run(cfg, data=None):
    """One run -> results row, appended to cfg.results_csv. data: {split: (x int16, y)} or load cache."""
    L.seed_everything(cfg.seed, workers=True, verbose=False)
    data = data or bd.load_eurosat(cfg.data_root)
    metrics = RUNNERS[cfg.mode](cfg, data)
    row = dict(backbone=cfg.backbone, init=cfg.init, mode=cfg.mode, frac=cfg.frac, seed=cfg.seed,
               in_chans=bm.in_chans_of(cfg.init), **metrics,
               config_hash=config_hash(cfg), config=json.dumps(recipe(cfg), sort_keys=True))
    log_row(cfg.results_csv, row)
    return row


def main(argv=None, data=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML (default: built-in = configs/default.yaml)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="config overrides")
    args = parser.parse_args(argv)
    row = run(load_config(args.config, args.set), data)
    print({key: (round(value, 4) if isinstance(value, float) else value) for key, value in row.items()
           if key != "config"})
    return row


if __name__ == "__main__":
    main()
