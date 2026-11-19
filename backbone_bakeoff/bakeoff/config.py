"""Run config: YAML file + `key=value` overrides -> RunConfig. Unknown keys / bad choices raise."""
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

# pod PVC if present, else parent dir (same rule as notebook)
WORK_DIR = "/home/jovyan/work" if os.path.isdir("/home/jovyan/work") else ".."
EPOCHS_BY_FRAC = {0.01: 200, 0.1: 50, 1.0: 15}  # same per fraction for every backbone
CHOICES = {"mode": ("ft", "lp"), "aug": ("d4", "none")}


@dataclass
class RunConfig:
    # run
    mode: str = "ft"
    backbone: str = "resnet50"
    init: str = "imagenet_rgb"
    frac: float = 0.1
    seed: int = 0
    # shared
    img_size: int = 224
    num_classes: int = 10
    pretrained: bool = True
    precision: str = "bf16-mixed"
    # FT recipe
    epochs: Optional[int] = None  # None -> epochs_by_frac[frac]
    epochs_by_frac: dict = field(default_factory=lambda: dict(EPOCHS_BY_FRAC))
    lr: float = 2e-4
    wd: float = 0.05
    warmup_frac: float = 0.05
    label_smooth: float = 0.1
    clip: float = 1.0
    bs: int = 64
    drop_rate: float = 0.0       # head dropout (timm)
    drop_path_rate: float = 0.0  # stochastic depth, linear over blocks (timm)
    aug: str = "d4"              # d4 = random flips + transpose; none
    val_every: int = 0           # >0: val acc curve every n epochs
    # LP recipe
    lp_epochs: int = 100
    lp_lr: float = 1e-3
    lp_bs: int = 256
    lp_wd_grid: list = field(default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2])
    # system (not in hash)
    out_dir: Optional[str] = None      # None -> WORK_DIR/saved_models/backbone_bakeoff (= notebook OUT_DIR)
    data_root: Optional[str] = None    # None -> WORK_DIR/data/eurosat (= notebook cache)
    results_csv: Optional[str] = None  # None -> out_dir/results.csv (= notebook)
    accelerator: str = "auto"
    num_workers: int = 0               # loader workers; data already in RAM
    progress: bool = True
    curves: bool = True                # FT: out_dir/runs/<hash>/metrics.csv
    tensorboard: bool = False          # FT: + TB events (needs tensorboard)

    @property
    def max_epochs(self):
        return self.epochs if self.epochs is not None else self.epochs_by_frac[self.frac]


SHARED_KEYS = {"mode", "backbone", "init", "frac", "seed", "img_size", "num_classes", "pretrained", "precision"}
SYSTEM_KEYS = {"out_dir", "data_root", "results_csv", "accelerator", "num_workers", "progress", "curves", "tensorboard"}
FT_KEYS = {"epochs", "epochs_by_frac", "lr", "wd", "warmup_frac", "label_smooth", "clip", "bs",
           "drop_rate", "drop_path_rate", "aug", "val_every"}
LP_KEYS = {"lp_epochs", "lp_lr", "lp_bs", "lp_wd_grid"}
MODE_KEYS = {"ft": FT_KEYS, "lp": LP_KEYS}
_groups = (SHARED_KEYS, SYSTEM_KEYS, FT_KEYS, LP_KEYS)  # each field in exactly one, else hash drifts silently
assert set().union(*_groups) == {f.name for f in dataclasses.fields(RunConfig)} and \
    sum(map(len, _groups)) == len(dataclasses.fields(RunConfig))


def recipe(cfg):
    """Fields defining the run for cfg.mode (shared + mode's own); epochs resolved."""
    keep = SHARED_KEYS | MODE_KEYS[cfg.mode]
    r = {k: v for k, v in dataclasses.asdict(cfg).items() if k in keep}
    if cfg.mode == "ft":
        del r["epochs_by_frac"]
        r["epochs"] = cfg.max_epochs
    return r


def config_hash(cfg):
    """Run id: sha1 of recipe() (incl. backbone/init/frac/seed)."""
    return hashlib.sha1(json.dumps(recipe(cfg), sort_keys=True).encode()).hexdigest()[:10]


def load_config(path=None, overrides=()):
    """YAML (optional) then overrides ["k=v", ...]; values parsed as YAML scalars."""
    d = {}
    if path:
        with open(path) as f:
            d.update(yaml.safe_load(f) or {})
    for o in overrides:
        k, _, v = o.partition("=")
        d[k.strip()] = yaml.safe_load(v)
    known = {f.name for f in dataclasses.fields(RunConfig)}
    if unknown := set(d) - known:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for f in dataclasses.fields(RunConfig):  # YAML reads "1e-3" as str
        t = f.type.__args__[0] if getattr(f.type, "__args__", None) else f.type  # Optional[int] -> int
        if f.name in d and d[f.name] is not None and t in (int, float):
            d[f.name] = t(d[f.name])
    cfg = RunConfig(**d)
    for k, ok in CHOICES.items():
        if getattr(cfg, k) not in ok:
            raise ValueError(f"{k}={getattr(cfg, k)!r} not in {ok}")
    cfg.out_dir = cfg.out_dir or os.path.join(WORK_DIR, "saved_models", "backbone_bakeoff")
    cfg.data_root = cfg.data_root or os.path.join(WORK_DIR, "data", "eurosat")
    cfg.results_csv = cfg.results_csv or os.path.join(cfg.out_dir, "results.csv")
    cfg.epochs_by_frac = {float(k): int(v) for k, v in cfg.epochs_by_frac.items()}
    cfg.lp_wd_grid = [float(w) for w in cfg.lp_wd_grid]
    return cfg
