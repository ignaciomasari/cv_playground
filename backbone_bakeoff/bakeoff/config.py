"""Run config: YAML file + `key=value` overrides -> RunConfig. Unknown keys / bad choices raise."""
import dataclasses
import hashlib
import json
import os
import types
import typing
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

import yaml

# pod PVC if present, else parent dir (same rule as notebook)
WORK_DIR = "/home/jovyan/work" if os.path.isdir("/home/jovyan/work") else ".."
EPOCHS_BY_FRAC = {0.01: 200, 0.1: 50, 1.0: 15}  # same per fraction for every backbone


# StrEnum: == / hash / json / f-string same as plain str -> config_hash + CSV unchanged
class Mode(StrEnum):
    FT = "ft"  # full fine-tune (Lightning)
    LP = "lp"  # frozen features + linear head


class Init(StrEnum):
    IMAGENET_RGB = "imagenet_rgb"    # ImageNet weights, RGB bands
    IMAGENET_13CH = "imagenet_13ch"  # ImageNet weights, stem tiled to 13 bands
    S2_SSL = "s2_ssl"                # SSL4EO-S12 MoCo weights, 13 bands

    @property
    def in_chans(self) -> int:
        return 3 if self is Init.IMAGENET_RGB else 13


class Aug(StrEnum):
    D4 = "d4"  # random flips + transpose
    NONE = "none"


@dataclass
class RunConfig:
    # run
    mode: Mode = Mode.FT
    backbone: str = "resnet50"
    init: Init = Init.IMAGENET_RGB
    frac: float = 0.1
    seed: int = 0
    # shared
    img_size: int = 224
    num_classes: int = 10
    pretrained: bool = True
    precision: str = "bf16-mixed"
    # FT recipe
    epochs: int | None = None  # None -> epochs_by_frac[frac]
    epochs_by_frac: dict[float, int] = field(default_factory=lambda: dict(EPOCHS_BY_FRAC))
    lr: float = 2e-4
    wd: float = 0.05
    warmup_frac: float = 0.05
    label_smooth: float = 0.1
    clip: float = 1.0
    bs: int = 64
    drop_rate: float = 0.0       # head dropout (timm)
    drop_path_rate: float = 0.0  # stochastic depth, linear over blocks (timm)
    aug: Aug = Aug.D4
    val_every: int = 0           # >0: val acc curve every n epochs
    # LP recipe
    lp_epochs: int = 100
    lp_lr: float = 1e-3
    lp_bs: int = 256
    lp_wd_grid: list[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2])
    # system (not in hash)
    out_dir: str | None = None      # None -> WORK_DIR/saved_models/backbone_bakeoff (= notebook OUT_DIR)
    data_root: str | None = None    # None -> WORK_DIR/data/eurosat (= notebook cache)
    results_csv: str | None = None  # None -> out_dir/results.csv (= notebook)
    accelerator: str = "auto"
    num_workers: int = 0            # loader workers; data already in RAM
    progress: bool = True
    curves: bool = True             # FT: out_dir/runs/<hash>/metrics.csv
    tensorboard: bool = False       # FT: + TB events (needs tensorboard)

    def __post_init__(self):
        """Coerce types (YAML "1e-3" is str), validate choices, resolve paths. Idempotent."""
        for config_field in dataclasses.fields(self):
            value, field_type = getattr(self, config_field.name), base_type(config_field.type)
            if value is None:
                continue
            if isinstance(field_type, type) and issubclass(field_type, StrEnum):
                try:
                    value = field_type(value)
                except ValueError:
                    raise ValueError(f"{config_field.name}={value!r} not in {[str(m) for m in field_type]}") from None
            elif field_type in (int, float):
                value = field_type(value)
            setattr(self, config_field.name, value)
        self.epochs_by_frac = {float(frac): int(epochs) for frac, epochs in self.epochs_by_frac.items()}
        self.lp_wd_grid = [float(wd) for wd in self.lp_wd_grid]
        self.out_dir = self.out_dir or os.path.join(WORK_DIR, "saved_models", "backbone_bakeoff")
        self.data_root = self.data_root or os.path.join(WORK_DIR, "data", "eurosat")
        self.results_csv = self.results_csv or os.path.join(self.out_dir, "results.csv")
        for key in ("out_dir", "data_root", "results_csv"):  # "{WORK_DIR}/..." -> pod/local work dir
            setattr(self, key, getattr(self, key).replace("{WORK_DIR}", WORK_DIR))

    @property
    def max_epochs(self) -> int:
        return self.epochs if self.epochs is not None else self.epochs_by_frac[self.frac]


def base_type(annotation: Any) -> Any:
    """`int | None` -> int; other annotations unchanged."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return next(arg for arg in typing.get_args(annotation) if arg is not type(None))
    return annotation


SHARED_KEYS = {"mode", "backbone", "init", "frac", "seed", "img_size", "num_classes", "pretrained", "precision"}
SYSTEM_KEYS = {"out_dir", "data_root", "results_csv", "accelerator", "num_workers", "progress", "curves", "tensorboard"}
FT_KEYS = {"epochs", "epochs_by_frac", "lr", "wd", "warmup_frac", "label_smooth", "clip", "bs",
           "drop_rate", "drop_path_rate", "aug", "val_every"}
LP_KEYS = {"lp_epochs", "lp_lr", "lp_bs", "lp_wd_grid"}
MODE_KEYS = {Mode.FT: FT_KEYS, Mode.LP: LP_KEYS}
_groups = (SHARED_KEYS, SYSTEM_KEYS, FT_KEYS, LP_KEYS)  # each field in exactly one, else hash drifts silently
assert set().union(*_groups) == {field.name for field in dataclasses.fields(RunConfig)} and \
    sum(map(len, _groups)) == len(dataclasses.fields(RunConfig))


def recipe(cfg: RunConfig) -> dict[str, Any]:
    """Fields defining the run for cfg.mode (shared + mode's own); epochs resolved."""
    keep = SHARED_KEYS | MODE_KEYS[cfg.mode]
    values = {key: value for key, value in dataclasses.asdict(cfg).items() if key in keep}
    if cfg.mode == Mode.FT:
        del values["epochs_by_frac"]
        values["epochs"] = cfg.max_epochs
    return values


def config_hash(cfg: RunConfig) -> str:
    """Run id: sha1 of recipe() (incl. backbone/init/frac/seed)."""
    return hashlib.sha1(json.dumps(recipe(cfg), sort_keys=True).encode()).hexdigest()[:10]


def load_config(path: str | None = None, overrides: Sequence[str] = ()) -> RunConfig:
    """YAML (optional) then overrides ["key=value", ...] (YAML values; text fields fall back to raw string)."""
    values = {}
    if path:
        with open(path) as file:
            values.update(yaml.safe_load(file) or {})
    fields = dataclasses.fields(RunConfig)
    str_keys = {field.name for field in fields
                if isinstance(base_type(field.type), type) and issubclass(base_type(field.type), str)}
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
    return RunConfig(**values)
