import numpy as np
import pytest

import mammo_cbm as M


def test_contributions_sum_to_logit(bundle, x):
    r = M.analyse(bundle, x)
    logit = np.log(r["p"] / (1 - r["p"]))
    assert logit == pytest.approx(bundle.base_logit + sum(r["contrib"].values()), abs=1e-4)


def test_override_sets_concept_and_changes_probs(bundle, x):
    r0, r1 = M.analyse(bundle, x), M.analyse(bundle, x, {"margins": 2})
    assert r1["probs"]["margins"].tolist() == [0, 0, 1]
    assert r0["p"] != r1["p"]


def test_ood_flag_and_verdict_values(bundle, x):
    r = M.analyse(bundle, x)
    assert r["verdict"] in {"Suspicious", "Likely benign", "Uncertain"}
    assert r["ood"] is False  # identity precision with a huge threshold


def test_gradcam_shape_and_range(bundle, x):
    cam = M.gradcam(bundle.model, x, "margins", 0)
    assert cam.shape == (M.IMG, M.IMG) and 0 <= cam.min() and cam.max() <= 1


def test_bundle_loads_with_weights_only_and_lists_concepts(bundle):
    assert bundle.names == [
        "lesion: CALC",
        "lesion: MASS",
        "margins: CIRCUMSCRIBED",
        "margins: NONE",
        "margins: SPICULATED",
    ]
