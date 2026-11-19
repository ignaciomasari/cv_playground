import numpy as np
import torch

from bakeoff.data import subset_idx

# 3 classes, sizes 10/20/30
Y = np.repeat(np.arange(3), [10, 20, 30])


def test_subset_is_stratified_per_class():
    idx = subset_idx(Y, frac=0.1, seed=0, num_classes=3)
    assert np.bincount(Y[idx], minlength=3).tolist() == [1, 2, 3]


def test_subset_keeps_at_least_one_per_class():
    idx = subset_idx(Y, frac=0.01, seed=0, num_classes=3)
    assert np.bincount(Y[idx], minlength=3).tolist() == [1, 1, 1]


def test_full_fraction_returns_all_indices():
    assert subset_idx(Y, frac=1.0, seed=2, num_classes=3).tolist() == list(range(60))


def test_subset_matches_notebook_subsets():
    # literals captured from original notebook impl -> paired comparisons stay valid
    assert subset_idx(Y, 0.1, 0, 3).tolist() == [1, 18, 19, 46, 51, 58]
    assert subset_idx(Y, 0.2, 1, 3).tolist() == [4, 7, 13, 16, 20, 27, 30, 37, 47, 52, 56, 59]


def test_subset_accepts_torch_labels():
    assert subset_idx(torch.from_numpy(Y), 0.1, 0, 3).tolist() == [1, 18, 19, 46, 51, 58]


# --- band order
import pytest

from bakeoff.data import RGB_IDX, SSL_ORDER, ssl_band_index

# torchgeo EuroSAT.all_band_names (file order, B8A last)
EUROSAT_FILE_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A")


def test_eurosat_bands_reordered_to_ssl4eo_order():
    # B8A (file pos 12) must land right after B08
    assert ssl_band_index(EUROSAT_FILE_ORDER) == [0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 9, 10, 11]


def test_reordered_names_follow_ssl4eo_s12():
    order = ssl_band_index(EUROSAT_FILE_ORDER)
    assert [EUROSAT_FILE_ORDER[i] for i in order] == [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


def test_rgb_idx_points_to_red_green_blue_in_cache():
    assert [SSL_ORDER[i] for i in RGB_IDX] == ["B04", "B03", "B02"]


def test_missing_band_raises():
    with pytest.raises(ValueError):
        ssl_band_index(EUROSAT_FILE_ORDER[:-1])
