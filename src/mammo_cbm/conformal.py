"""Class-conditional conformal thresholds and the final verdict."""

import numpy as np


def conformal_thresholds(p, y, alpha=0.10):
    """Per-class thresholds [q_benign, q_malignant] from calibration data (p = P(malignant), y in {0, 1}).

    Class k stays in the prediction set when its nonconformity score (1 - P(class k)) <= q_k, which keeps
    about (1 - alpha) of true cases of each class correctly labelled.
    """
    p, y = np.asarray(p, float), np.asarray(y)

    def qhat(s):
        n = len(s)
        return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))

    return np.array([qhat(p[y == 0]), qhat(1 - p[y == 1])])


def verdict_from_p(p, q):
    """Map P(malignant) to Suspicious / Likely benign / Uncertain using the conformal thresholds q."""
    keep = np.array([p, 1 - p]) <= q  # scores: benign = p, malignant = 1 - p
    if keep[1] and not keep[0]:
        return "Suspicious"
    if keep[0] and not keep[1]:
        return "Likely benign"
    return "Uncertain"
