"""Backbone registry, per-init model building, GPU-side input prep."""
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Init, RunConfig
from .data import RGB_DN_MAX, RGB_IDX, band_stats


@dataclass(frozen=True)
class Backbone:
    arch: str                # timm arch
    imagenet_tag: str        # timm pretrained tag
    s2_weights: str | None   # torchgeo "Enum.MEMBER", resolved lazily
    inits: tuple[Init, ...]  # supported inits

    @property
    def imagenet_name(self) -> str:
        return f"{self.arch}.{self.imagenet_tag}"

    def supports(self, init: str) -> bool:
        return init in self.inits


BACKBONES = {
    "resnet50": Backbone("resnet50", "a1_in1k", "ResNet50_Weights.SENTINEL2_ALL_MOCO",
                         (Init.IMAGENET_RGB, Init.S2_SSL)),
    "vit_s16": Backbone("vit_small_patch16_224", "augreg_in21k_ft_in1k", "ViTSmall16_Weights.SENTINEL2_ALL_MOCO",
                        (Init.IMAGENET_RGB, Init.S2_SSL)),
    "convnext_t": Backbone("convnext_tiny", "fb_in22k", None,  # no S2-SSL in torchgeo
                           (Init.IMAGENET_RGB, Init.IMAGENET_13CH)),
}
# keys allowed missing/unexpected when loading S2-SSL backbones (head + ViT final norm)
S2_OK_KEYS = {"fc.weight", "fc.bias", "head.weight", "head.bias", "norm.weight", "norm.bias"}


def available(backbone: str, init: str) -> bool:
    """Unknown backbone raises (typo in grid != silently dropped)."""
    if backbone not in BACKBONES:
        raise ValueError(f"backbone={backbone!r} not in {list(BACKBONES)}")
    return BACKBONES[backbone].supports(init)


@dataclass
class InputSpec:
    """How raw int16 DN become model input: band idx, /scale, optional clip to [0,1], (x-mean)/std."""
    idx: list[int]
    scale: float
    clip: bool
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self):
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.as_tensor(self.std, dtype=torch.float32).view(1, -1, 1, 1)

    def to(self, device: torch.device | str) -> "InputSpec":
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self


def _s2_weights(name: str):
    import torchgeo.models as tgm
    enum, member = name.split(".")
    return getattr(getattr(tgm, enum), member)


def load_s2_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load S2-SSL backbone; only head / final-norm keys may differ."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad = (set(missing) | set(unexpected)) - S2_OK_KEYS
    if bad:
        raise ValueError(f"S2 checkpoint key mismatch: {sorted(bad)[:5]}...")


def build(backbone: str, init: Init | str, num_classes: int = 10, *, device: torch.device | str = "cpu",
          band_mean: torch.Tensor | None = None, band_std: torch.Tensor | None = None, pretrained: bool = True,
          drop_rate: float = 0.0, drop_path_rate: float = 0.0) -> tuple[nn.Module, InputSpec]:
    """-> (model on device, channels_last; InputSpec). band_mean/std needed for imagenet_13ch."""
    init = Init(init)
    if not available(backbone, init):
        raise ValueError(f"{init} not available for {backbone}")
    entry = BACKBONES[backbone]
    timm_kwargs = dict(num_classes=num_classes, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
    if init is Init.IMAGENET_RGB:
        model = timm.create_model(entry.imagenet_name, pretrained=pretrained, **timm_kwargs)
        spec = InputSpec(RGB_IDX, scale=RGB_DN_MAX, clip=True,
                         mean=model.pretrained_cfg["mean"], std=model.pretrained_cfg["std"])
    elif init is Init.IMAGENET_13CH:
        if band_mean is None or band_std is None:
            raise ValueError("imagenet_13ch needs band_mean/band_std")
        # timm adapt_input_conv: stem filters tiled R,G,B,... over 13 bands, x 3/13
        model = timm.create_model(entry.imagenet_name, pretrained=pretrained, in_chans=13, **timm_kwargs)
        spec = InputSpec(list(range(13)), scale=1.0, clip=False, mean=band_mean, std=band_std)
    else:  # Init.S2_SSL
        model = timm.create_model(entry.arch, in_chans=13, **timm_kwargs)
        if pretrained:
            load_s2_weights(model, _s2_weights(entry.s2_weights).get_state_dict(progress=True))
        spec = InputSpec(list(range(13)), scale=10_000.0, clip=False, mean=torch.zeros(13), std=torch.ones(13))
    return model.to(device, memory_format=torch.channels_last), spec.to(device)


def build_from(cfg: RunConfig, train_x: torch.Tensor, *, num_classes: int | None = None,
               device: torch.device | str = "cpu") -> tuple[nn.Module, InputSpec]:
    """build() from RunConfig; train_x (int16) only used for imagenet_13ch band stats."""
    mean, std = band_stats(train_x) if cfg.init is Init.IMAGENET_13CH else (None, None)
    return build(cfg.backbone, cfg.init, cfg.num_classes if num_classes is None else num_classes, device=device,
                 band_mean=mean, band_std=std, pretrained=cfg.pretrained,
                 drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate)


def prep(x_batch: torch.Tensor, spec: InputSpec, train: bool, img_size: int = 224) -> torch.Tensor:
    """int16 batch -> normalised float batch at img_size on spec's device; train: random D4 aug."""
    device = spec.mean.device
    x = x_batch.to(device, non_blocking=True)[:, spec.idx].float() / spec.scale
    if spec.clip:
        x = x.clamp(0, 1)
    if train:  # per-sample hflip, vflip, transpose = all 8 dihedral variants
        batch_size = x.size(0)
        for transform in (lambda img: img.flip(-1), lambda img: img.flip(-2), lambda img: img.transpose(-1, -2)):
            applies = torch.rand(batch_size, device=device) < 0.5
            x = torch.where(applies.view(batch_size, 1, 1, 1), transform(x), x)
    if x.shape[-1] != img_size or x.shape[-2] != img_size:
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
    x = (x - spec.mean) / spec.std
    return x.contiguous(memory_format=torch.channels_last)
