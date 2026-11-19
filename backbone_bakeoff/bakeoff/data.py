"""EuroSAT data: band order, int16 cache, stratified subsets, batching."""
import numpy as np
import torch

# SSL4EO-S12 order (torchgeo _sentinel2_toa_bands); EuroSAT files store B8A last
SSL_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
RGB_IDX = [SSL_ORDER.index(b) for b in ("B04", "B03", "B02")]
RGB_DN_MAX = 3000.0  # L1C DN -> [0,1] for RGB/ImageNet path (model input + previews)


def ssl_band_index(src_names):
    """Indices into src_names giving SSL4EO band order."""
    missing = set(SSL_ORDER) - set(src_names)
    if missing:
        raise ValueError(f"missing bands: {sorted(missing)}")
    return [list(src_names).index(b) for b in SSL_ORDER]


def subset_idx(y, frac, seed, num_classes=10):
    """Stratified train subset; same (frac, seed) -> same idx for every model (paired comparison)."""
    y = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    if frac >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(10_000 + seed)
    idx = [rng.choice(np.flatnonzero(y == c), max(1, round(frac * (y == c).sum())), replace=False)
           for c in range(num_classes)]
    return np.sort(np.concatenate(idx))


def cache_split(root, split, num_workers=4):
    """EuroSAT split -> (int16 x[N,13,64,64] in SSL order, long y); cached as .npy under root."""
    import os
    xp, yp = (os.path.join(root, f"cache_{split}_{k}.npy") for k in ("x", "y"))
    if not os.path.isfile(xp):
        from torchgeo.datasets import EuroSAT  # heavy: only for first build
        from tqdm.auto import tqdm
        ds = EuroSAT(root, split=split, bands=EuroSAT.all_band_names, download=True)
        order = ssl_band_index(EuroSAT.all_band_names)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=num_workers)
        xs, ys = [], []
        for b in tqdm(dl, desc=f"cache {split}"):
            xs.append(b["image"][:, order].round().clamp(0, 32767).to(torch.int16).numpy())
            ys.append(b["label"].numpy())
        np.save(xp, np.concatenate(xs)); np.save(yp, np.concatenate(ys))
    return torch.from_numpy(np.load(xp)), torch.from_numpy(np.load(yp)).long()


def load_eurosat(root, num_workers=4):
    """{split: (x, y)} for train/val/test."""
    return {s: cache_split(root, s, num_workers) for s in ("train", "val", "test")}


def class_names(root):
    import os
    from torchgeo.datasets import EuroSAT
    return sorted(os.listdir(os.path.join(root, EuroSAT.base_dir)))


EVAL_BS = 256  # eval / feature extraction batch


def train_subset(data, cfg):
    """-> (idx, x, y) of cfg.frac stratified train subset for cfg.seed."""
    idx = torch.from_numpy(subset_idx(data["train"][1], cfg.frac, cfg.seed, cfg.num_classes))
    return idx, data["train"][0][idx], data["train"][1][idx]


def band_stats(x, n=2000):
    """Per-band DN mean/std on first n samples (for imagenet_13ch z-score)."""
    xt = x[:n].float()
    return xt.mean((0, 2, 3)), xt.std((0, 2, 3))


def iterate(n, bs, shuffle, drop_last=False):
    """Index batches; drop_last only if n >= bs."""
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    stop = n - n % bs if (drop_last and n >= bs) else n
    for i in range(0, stop, bs):
        yield idx[i:i + bs]
