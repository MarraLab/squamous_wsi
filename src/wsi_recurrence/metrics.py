import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


def compute_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def compute_pr_auc(y_true, y_pred):
    return average_precision_score(y_true, y_pred)


def compute_roc(y_true, y_pred):
    return roc_curve(y_true, y_pred)


def bootstrap_roc(y_true, y_pred, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        fpr, tpr, _ = roc_curve(y_true[idx], y_pred[idx])
        tprs.append(np.interp(base_fpr, fpr, tpr))

    return base_fpr, np.percentile(tprs, [2.5, 97.5], axis=0)


def plot_roc(ax, y_true, y_pred, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    base_fpr, (tpr_lo, tpr_hi) = bootstrap_roc(y_true, y_pred)

    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc_score:.3f})" if label else f"AUC={auc_score:.3f}")
    ax.fill_between(base_fpr, tpr_lo, tpr_hi, alpha=0.2)

    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.grid(alpha=0.3)

    return auc_score


def plot_pr(ax, y_true, y_pred, label=None):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    ax.plot(
        recall,
        precision,
        lw=2,
        label=f"{label} (AP={pr_auc:.3f})" if label else f"AP={pr_auc:.3f}",
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)

    return pr_auc
