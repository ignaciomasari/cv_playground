import pytest
import torch

from bakeoff.data import RGB_IDX
from bakeoff.models import InputSpec, build, load_s2_weights, prep


def first_conv_in_chans(m):
    return next(mod for mod in m.modules() if isinstance(mod, torch.nn.Conv2d)).in_channels


# --- prep

S2 = InputSpec(idx=list(range(13)), scale=10_000.0, clip=False, mean=torch.zeros(13), std=torch.ones(13))


def test_prep_s2_scales_dn_to_reflectance_and_resizes():
    x = torch.full((2, 13, 4, 4), 5000, dtype=torch.int16)
    out = prep(x, S2, train=False, img_size=8)
    assert out.shape == (2, 13, 8, 8)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_prep_rgb_selects_red_green_blue_and_clips():
    x = torch.zeros(1, 13, 4, 4, dtype=torch.int16)
    x[:, 3], x[:, 2], x[:, 1] = 6000, 1500, 0  # B04 over-range, B03 half, B02 zero
    spec = InputSpec(idx=RGB_IDX, scale=3000.0, clip=True, mean=torch.full((3,), 0.5), std=torch.full((3,), 0.5))
    out = prep(x, spec, train=False, img_size=4)
    assert out[0, :, 0, 0].tolist() == [1.0, 0.0, -1.0]  # (clip(v/3000)-.5)/.5


def test_prep_zscore_per_band():
    x = torch.zeros(1, 13, 4, 4, dtype=torch.int16)
    x[:, 0] = 300
    spec = InputSpec(idx=list(range(13)), scale=1.0, clip=False, mean=torch.full((13,), 100.0), std=torch.full((13,), 50.0))
    out = prep(x, spec, train=False, img_size=4)
    assert out[0, 0, 0, 0].item() == 4.0 and out[0, 1, 0, 0].item() == -2.0


def test_prep_train_aug_only_rearranges_pixels():
    torch.manual_seed(0)
    x = torch.randint(0, 10_000, (16, 13, 4, 4), dtype=torch.int16)
    ev, tr = prep(x, S2, train=False, img_size=4), prep(x, S2, train=True, img_size=4)
    assert torch.equal(ev.flatten(2).sort(-1).values, tr.flatten(2).sort(-1).values)
    assert not torch.equal(ev, tr)  # 16 samples: some flipped


def test_prep_eval_is_deterministic():
    x = torch.randint(0, 10_000, (4, 13, 4, 4), dtype=torch.int16)
    assert torch.equal(prep(x, S2, False, 8), prep(x, S2, False, 8))


# --- build (no downloads)

@pytest.mark.parametrize("bb,init,chans", [
    ("resnet50", "imagenet_rgb", 3), ("resnet50", "s2_ssl", 13),
    ("convnext_t", "imagenet_13ch", 13), ("vit_s16", "s2_ssl", 13)])
def test_build_input_channels_follow_init(bb, init, chans):
    m, spec = build(bb, init, pretrained=False, band_mean=torch.zeros(13), band_std=torch.ones(13))
    assert first_conv_in_chans(m) == chans == len(spec.idx)


def test_build_head_has_num_classes():
    m, _ = build("resnet50", "s2_ssl", num_classes=10, pretrained=False)
    assert m(torch.zeros(1, 13, 64, 64)).shape == (1, 10)


def test_imagenet_rgb_spec_uses_each_models_pretrain_norm():
    _, rn = build("resnet50", "imagenet_rgb", pretrained=False)
    _, vit = build("vit_s16", "imagenet_rgb", pretrained=False)
    assert rn.idx == RGB_IDX and rn.clip
    assert torch.allclose(rn.mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    assert torch.allclose(vit.mean.flatten(), torch.tensor([0.5, 0.5, 0.5]))  # augreg != ImageNet stats


def test_imagenet_13ch_spec_is_band_zscore():
    mean, std = torch.arange(13.0), torch.arange(1.0, 14.0)
    _, spec = build("convnext_t", "imagenet_13ch", pretrained=False, band_mean=mean, band_std=std)
    assert torch.equal(spec.mean.flatten(), mean) and torch.equal(spec.std.flatten(), std)


def test_s2_spec_divides_by_10k():
    _, spec = build("vit_s16", "s2_ssl", pretrained=False)
    assert spec.scale == 10_000.0 and not spec.clip


def test_unavailable_init_raises():
    with pytest.raises(ValueError):
        build("convnext_t", "s2_ssl", pretrained=False)


# --- S2 checkpoint key guard

def test_s2_weights_load_when_only_head_missing():
    m, _ = build("resnet50", "s2_ssl", pretrained=False)
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("fc.")}
    load_s2_weights(m, sd)


def test_s2_weights_reject_missing_backbone_keys():
    m, _ = build("resnet50", "s2_ssl", pretrained=False)
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("layer4.")}
    with pytest.raises(ValueError):
        load_s2_weights(m, sd)
