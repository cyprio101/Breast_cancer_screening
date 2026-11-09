import numpy as np
import pytest
import torch

import mammo_cbm as M

HEADS = {"lesion": ["CALC", "MASS"], "margins": ["CIRCUMSCRIBED", "NONE", "SPICULATED"]}


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    """Tiny random model with a random decision layer: tests logic, not accuracy."""
    torch.manual_seed(0)
    model = M.CBM(HEADS, pretrained=False).eval()
    n = sum(len(v) for v in HEADS.values())
    rng = np.random.default_rng(0)
    d = model.backbone.num_features
    path = tmp_path_factory.mktemp("b") / "bundle.pt"
    M.save_bundle(
        path,
        model,
        HEADS,
        "efficientnet_b0",
        rng.normal(size=n),
        0.1,
        np.full(n, 0.3),
        [0.2, 0.2],
        np.zeros(d),
        np.eye(d),
        1e12,
    )
    return M.load_bundle(path)


@pytest.fixture()
def x():
    rng = np.random.default_rng(1)
    return M.normalise(M.prep_gray(rng.random((300, 300)).astype(np.float32) * 1000))
