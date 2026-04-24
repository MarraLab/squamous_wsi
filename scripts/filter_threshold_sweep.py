# %%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# load labeled tiles with predictions
df = pd.read_csv("all_tile_features.csv")

# if not already present, create y
df = df[df["label"].isin(["bad_line", "good_tissue"])].copy()
df["y"] = (df["label"] == "bad_line").astype(int)


import joblib
bundle = joblib.load("/projects/marralab/rcorbett_prj/LUSC/tile_filter_model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
df["bad_prob"] = model.predict_proba(df[FEATURES].values)[:, 1]


thresholds = np.arange(0.05, 0.96, 0.05)

rows = []
for thr in thresholds:
    pred_bad = (df["bad_prob"] >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(df["y"], pred_bad, labels=[0, 1]).ravel()

    bad_recall = tp / (tp + fn) if (tp + fn) else np.nan
    good_specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision_removed = tp / (tp + fp) if (tp + fp) else np.nan
    frac_removed = pred_bad.mean()

    rows.append({
        "threshold": thr,
        "bad_recall": bad_recall,
        "good_specificity": good_specificity,
        "precision_removed": precision_removed,
        "frac_removed": frac_removed,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    })

sweep_df = pd.DataFrame(rows)
sweep_df
# %%
# %%
per_slide_rows = []

for thr in thresholds:
    for slide_id, d in df.groupby("slide_id"):
        pred_bad = (d["bad_prob"] >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(d["y"], pred_bad, labels=[0, 1]).ravel()

        bad_recall = tp / (tp + fn) if (tp + fn) else np.nan
        good_specificity = tn / (tn + fp) if (tn + fp) else np.nan
        precision_removed = tp / (tp + fp) if (tp + fp) else np.nan
        frac_removed = pred_bad.mean()

        per_slide_rows.append({
            "slide_id": slide_id,
            "threshold": thr,
            "bad_recall": bad_recall,
            "good_specificity": good_specificity,
            "precision_removed": precision_removed,
            "frac_removed": frac_removed,
        })

per_slide_df = pd.DataFrame(per_slide_rows)
per_slide_df.head()
# %%
# %%
summary = (
    per_slide_df.groupby("threshold")
    .agg(
        min_bad_recall=("bad_recall", "min"),
        median_bad_recall=("bad_recall", "median"),
        min_good_specificity=("good_specificity", "min"),
        median_good_specificity=("good_specificity", "median"),
        max_frac_removed=("frac_removed", "max"),
        median_frac_removed=("frac_removed", "median"),
    )
    .reset_index()
)

summary
# %%
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sweep_df["threshold"], sweep_df["bad_recall"], label="bad recall")
ax.plot(sweep_df["threshold"], sweep_df["good_specificity"], label="good specificity")
ax.plot(sweep_df["threshold"], sweep_df["precision_removed"], label="precision removed")
ax.plot(sweep_df["threshold"], sweep_df["frac_removed"], label="fraction removed")
ax.set_xlabel("bad_prob threshold")
ax.set_ylabel("metric")
ax.legend()
ax.set_title("Threshold sweep")
plt.tight_layout()
plt.show()