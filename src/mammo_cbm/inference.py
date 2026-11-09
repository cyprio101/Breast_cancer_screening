"""Model bundle (safe to load) and the exact, additive explanation."""

import numpy as np
import torch
import torch.nn.functional as F

from .conformal import verdict_from_p
from .model import CBM


class Bundle:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def save_bundle(path, model, heads, backbone, coef, intercept, c_mean, q, ood_mu, ood_prec, ood_thr, meta=None):
    """Stores tensors, numbers and strings only, so torch.load(weights_only=True) can read it without running code."""

    def t(v):
        return torch.as_tensor(np.asarray(v), dtype=torch.float32)

    torch.save(
        dict(
            state=model.state_dict(),
            heads=heads,
            backbone=backbone,
            coef=t(coef),
            intercept=float(intercept),
            c_mean=t(c_mean),
            q=t(q),
            ood_mu=t(ood_mu),
            ood_prec=t(ood_prec),
            ood_thr=float(ood_thr),
            meta=meta or {},
        ),
        path,
    )


def load_bundle(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    model = CBM(d["heads"], d["backbone"], pretrained=False)
    model.load_state_dict(d["state"])
    model.eval()
    names = [f"{h}: {c}" for h, cl in d["heads"].items() for c in cl]
    coef, c_mean = d["coef"].numpy(), d["c_mean"].numpy()
    return Bundle(
        model=model,
        heads=d["heads"],
        names=names,
        coef=coef,
        c_mean=c_mean,
        base_logit=d["intercept"] + float(coef @ c_mean),
        q=d["q"].numpy(),
        ood_mu=d["ood_mu"].numpy(),
        ood_prec=d["ood_prec"].numpy(),
        ood_thr=d["ood_thr"],
        meta=d["meta"],
    )


def analyse(b, x, overrides=None):
    """P(malignant), verdict, exact per-concept contributions (log-odds vs the average training lesion),
    per-head probabilities and an out-of-distribution flag. overrides = {head: class_index} from the clinician."""
    with torch.no_grad():
        logits, g = b.model(x[None])
    probs = {k: F.softmax(v, 1)[0].numpy() for k, v in logits.items()}
    for k, i in (overrides or {}).items():
        probs[k] = np.eye(len(probs[k]), dtype=np.float32)[i]
    c = np.concatenate([probs[k] for k in b.heads])
    contrib = b.coef * (c - b.c_mean)
    p = float(1 / (1 + np.exp(-(b.base_logit + contrib.sum()))))
    d = g[0].numpy() - b.ood_mu
    return dict(
        p=p,
        verdict=verdict_from_p(p, b.q),
        contrib=dict(zip(b.names, contrib, strict=True)),
        probs=probs,
        base_p=float(1 / (1 + np.exp(-b.base_logit))),
        ood=bool(d @ b.ood_prec @ d > b.ood_thr),
    )
