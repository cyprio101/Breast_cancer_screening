import numpy as np
from PIL import Image

import mammo_cbm as M


def test_roi_crop_is_square_and_padded_at_edges():
    a = np.arange(100 * 120, dtype=np.float32).reshape(100, 120)
    c = M.roi_crop(a, (0, 0, 20, 20))  # box in the corner: crop must still be square
    assert c.shape[0] == c.shape[1]


def test_prep_gray_range_and_shape():
    g = M.prep_gray(np.random.default_rng(0).random((200, 150)).astype(np.float32) * 4000)
    assert g.shape == (M.IMG, M.IMG) and g.dtype == np.float32
    assert 0.0 <= g.min() and g.max() <= 1.0


def test_normalise_shape():
    assert M.normalise(np.zeros((M.IMG, M.IMG), np.float32)).shape == (3, M.IMG, M.IMG)


def test_load_png_8bit_and_16bit(tmp_path):
    a8 = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    Image.fromarray(a8).save(tmp_path / "a.png")
    assert M.load_array(tmp_path / "a.png").shape == (64, 64)
    a16 = (np.random.default_rng(0).random((64, 64)) * 60000).astype(np.uint16)
    Image.fromarray(a16).save(tmp_path / "b.png")
    assert M.load_array(tmp_path / "b.png").max() > 255  # full range kept, not squashed to 8 bit
