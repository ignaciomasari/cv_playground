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
assert set().union(*_groups) == {field.name for field in dataclasses.fields(RunConfig)} and \
    sum(map(len, _groups)) == len(dataclasses.fields(RunConfig))


def recipe(cfg):
    """Fields defining the run for cfg.mode (shared + mode's own); epochs resolved."""
    keep = SHARED_KEYS | MODE_KEYS[cfg.mode]
    values = {key: value for key, value in dataclasses.asdict(cfg).items() if key in keep}
    if cfg.mode == "ft":
        del values["epochs_by_frac"]
        values["epochs"] = cfg.max_epochs
    return values


def config_hash(cfg):
    """Run id: sha1 of recipe() (incl. backbone/init/frac/seed)."""
    return hashlib.sha1(json.dumps(recipe(cfg), sort_keys=True).encode()).hexdigest()[:10]


def load_config(path=None, overrides=()):
    """YAML (optional) then overrides ["key=value", ...] (YAML values; text fields fall back to raw string)."""
    values = {}
    if path:
        with open(path) as file:
            values.update(yaml.safe_load(file) or {})
    fields = dataclasses.fields(RunConfig)
    str_keys = {field.name for field in fields if field.type in (str, Optional[str])}
    for override in overrides:
        key, _, raw = override.partition("=")
        key = key.strip()
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        if key in str_keys and not isinstance(value, (str, type(None))):  # "{WORK_DIR}/x" -> dict, "16" -> int
            value = raw.strip()
        values[key] = value
    if unknown := set(values) - {field.name for field in fields}:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for field in fields:  # YAML reads "1e-3" as str
        field_type = field.type.__args__[0] if getattr(field.type, "__args__", None) else field.type  # Optional[int] -> int
        if field.name in values and values[field.name] is not None and field_type in (int, float):
            values[field.name] = field_type(values[field.name])
    cfg = RunConfig(**values)
    for key, allowed in CHOICES.items():
        if getattr(cfg, key) not in allowed:
            raise ValueError(f"{key}={getattr(cfg, key)!r} not in {allowed}")
    cfg.out_dir = cfg.out_dir or os.path.join(WORK_DIR, "saved_models", "backbone_bakeoff")
    cfg.data_root = cfg.data_root or os.path.join(WORK_DIR, "data", "eurosat")
    cfg.results_csv = cfg.results_csv or os.path.join(cfg.out_dir, "results.csv")
    for key in ("out_dir", "data_root", "results_csv"):  # "{WORK_DIR}/..." -> pod/local work dir
        setattr(cfg, key, getattr(cfg, key).replace("{WORK_DIR}", WORK_DIR))
    cfg.epochs_by_frac = {float(frac): int(epochs) for frac, epochs in cfg.epochs_by_frac.items()}
    cfg.lp_wd_grid = [float(wd) for wd in cfg.lp_wd_grid]
    return cfg
