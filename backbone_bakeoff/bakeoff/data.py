"""EuroSAT data: band order, int16 cache, stratified subsets, batching."""
import numpy as np
import torch

# SSL4EO-S12 order (torchgeo _sentinel2_toa_bands); EuroSAT files store B8A last
SSL_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
RGB_IDX = [SSL_ORDER.index(band) for band in ("B04", "B03", "B02")]
RGB_DN_MAX = 3000.0  # L1C DN -> [0,1] for RGB/ImageNet path (model input + previews)


def ssl_band_index(src_names):
    """Indices into src_names giving SSL4EO band order."""
    missing = set(SSL_ORDER) - set(src_names)
    if missing:
        raise ValueError(f"missing bands: {sorted(missing)}")
    return [list(src_names).index(band) for band in SSL_ORDER]


def subset_idx(y, frac, seed, num_classes=10):
    """Stratified train subset; same (frac, seed) -> same idx for every model (paired comparison)."""
    y = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    if frac >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(10_000 + seed)
    per_class = [rng.choice(np.flatnonzero(y == label), max(1, round(frac * (y == label).sum())), replace=False)
                 for label in range(num_classes)]
    return np.sort(np.concatenate(per_class))


def cache_split(root, split, num_workers=4):
    """EuroSAT split -> (int16 x[N,13,64,64] in SSL order, long y); cached as .npy under root."""
    import os
    x_path, y_path = (os.path.join(root, f"cache_{split}_{name}.npy") for name in ("x", "y"))
    if not os.path.isfile(x_path):
        from torchgeo.datasets import EuroSAT  # heavy: only for first build
        from tqdm.auto import tqdm
        dataset = EuroSAT(root, split=split, bands=EuroSAT.all_band_names, download=True)
        order = ssl_band_index(EuroSAT.all_band_names)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=num_workers)
        x_batches, y_batches = [], []
        for batch in tqdm(loader, desc=f"cache {split}"):
            x_batches.append(batch["image"][:, order].round().clamp(0, 32767).to(torch.int16).numpy())
            y_batches.append(batch["label"].numpy())
        np.save(x_path, np.concatenate(x_batches))
        np.save(y_path, np.concatenate(y_batches))
    return torch.from_numpy(np.load(x_path)), torch.from_numpy(np.load(y_path)).long()


def load_eurosat(root, num_workers=4):
    """{split: (x, y)} for train/val/test."""
    return {split: cache_split(root, split, num_workers) for split in ("train", "val", "test")}


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
    x_float = x[:n].float()
    return x_float.mean((0, 2, 3)), x_float.std((0, 2, 3))


def iterate(n, bs, shuffle, drop_last=False):
    """Index batches; drop_last only if n >= bs."""
    order = torch.randperm(n) if shuffle else torch.arange(n)
    stop = n - n % bs if (drop_last and n >= bs) else n
    for start in range(0, stop, bs):
        yield order[start:start + bs]
