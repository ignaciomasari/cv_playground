"""Run config: YAML file + `key=value` overrides -> FTConfig. Unknown keys raise."""
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

# pod PVC if present, else parent dir (same rule as notebook)
WORK_DIR = "/home/jovyan/work" if os.path.isdir("/home/jovyan/work") else ".."
EPOCHS_BY_FRAC = {0.01: 200, 0.1: 50, 1.0: 15}  # same per fraction for every backbone


@dataclass
class FTConfig:
    # run
    backbone: str = "resnet50"
    init: str = "imagenet_rgb"
    frac: float = 0.1
    seed: int = 0
    # recipe
    epochs: Optional[int] = None  # None -> epochs_by_frac[frac]
    epochs_by_frac: dict = field(default_factory=lambda: dict(EPOCHS_BY_FRAC))
    lr: float = 2e-4
    wd: float = 0.05
    warmup_frac: float = 0.05
    label_smooth: float = 0.1
    clip: float = 1.0
    bs: int = 64
    img_size: int = 224
    num_classes: int = 10
    pretrained: bool = True
    # system
    data_root: Optional[str] = None    # None -> WORK_DIR/data/eurosat (= notebook cache)
    results_csv: Optional[str] = None  # None -> WORK_DIR/saved_models/backbone_bakeoff/results.csv (= notebook)
    accelerator: str = "auto"
    precision: str = "bf16-mixed"
    num_workers: int = 0  # loader workers; data already in RAM
    progress: bool = True

    @property
    def max_epochs(self):
        return self.epochs if self.epochs is not None else self.epochs_by_frac[self.frac]


def load_config(path=None, overrides=()):
    """YAML (optional) then overrides ["k=v", ...]; values parsed as YAML scalars."""
    d = {}
    if path:
        with open(path) as f:
            d.update(yaml.safe_load(f) or {})
    for o in overrides:
        k, _, v = o.partition("=")
        d[k.strip()] = yaml.safe_load(v)
    known = {f.name for f in dataclasses.fields(FTConfig)}
    if unknown := set(d) - known:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = FTConfig(**d)
    cfg.data_root = cfg.data_root or os.path.join(WORK_DIR, "data", "eurosat")
    cfg.results_csv = cfg.results_csv or os.path.join(WORK_DIR, "saved_models", "backbone_bakeoff", "results.csv")
    cfg.epochs_by_frac = {float(k): int(v) for k, v in cfg.epochs_by_frac.items()}
    return cfg
