# %%
#!/usr/bin/env python
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt

from pathlib import Path
import io
import zipfile
import re
import glob
import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel

from wsi_recurrence.tile_filter.features import compute_features as compute_features_base
from wsi_recurrence.tile_filter.zip_tiles import build_zip_coord_index, match_tile_name

# -------------------------
# Edit these
# -------------------------
LABEL_CSV = Path("tile_labels/combined.csv")
CACHE_DIR = Path("/tmp/image_cache")
OUT_CSV = Path("all_tile_features.csv")

# Coordinate matching tolerance in microns
COORD_TOL = 2.0

# -------------------------
# Helpers
# -------------------------
def compute_features(img_pil: Image.Image) -> dict:
    feats = compute_features_base(img_pil)

    img = np.array(img_pil.convert("RGB"))
    gray = rgb2gray(img)
    hsv = rgb2hsv(img)

    feats.update(
        {
            "p10_intensity": float(np.percentile(gray, 10)),
            "p90_intensity": float(np.percentile(gray, 90)),
            "dark_frac": float((gray < 0.35).mean()),
            "bright_frac": float((gray > 0.90).mean()),
            "sat_std": float(hsv[..., 1].std()),
            "edge_std": float(sobel(gray).std()),
        }
    )
    return feats


def find_zip_for_slide(slide_id: str, cache_dir: Path):
    matches = sorted(cache_dir.glob(f"{slide_id}.*.zip"))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"WARNING: multiple zips for {slide_id}, using first:")
        for m in matches:
            print(" ", m)
    return matches[0]


# -------------------------
# Load labels
# -------------------------
labels_df = pd.read_csv(LABEL_CSV)
labels_df = labels_df[labels_df["label"].isin(["bad_line", "good_tissue"])].copy()

print("Overall label counts:")
print(labels_df["label"].value_counts())
print("\nSlides:")
print(labels_df["slide_id"].value_counts())

# -------------------------
# Process all slides
# -------------------------
all_rows = []
all_missing = []
slide_summaries = []

for slide_id, slide_df in labels_df.groupby("slide_id"):
    print(f"\n--- Processing {slide_id} ---")
    zip_path = find_zip_for_slide(slide_id, CACHE_DIR)

    if zip_path is None:
        print(f"WARNING: no zip found for {slide_id}")
        all_missing.extend([(slide_id, int(r.tile_idx), float(r.x_um), float(r.y_um), "no_zip")
                            for _, r in slide_df.iterrows()])
        continue

    print(f"Using zip: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_df = build_zip_coord_index(zf.namelist())
        print(f"Indexed {len(zip_df)} tile jpgs from zip")

        matched_rows = 0
        missing_rows = 0
        dists = []

        for _, row in slide_df.iterrows():
            x_um = float(row["x_um"])
            y_um = float(row["y_um"])
            tile_idx = int(row["tile_idx"])
            label = row["label"]

            tile_name, dist = match_tile_name(x_um, y_um, zip_df, tol=COORD_TOL)

            if tile_name is None:
                missing_rows += 1
                all_missing.append((slide_id, tile_idx, x_um, y_um, "no_coord_match"))
                continue

            img_bytes = zf.read(tile_name)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            feats = compute_features(img)
            feats.update({
                "slide_id": slide_id,
                "tile_idx": tile_idx,
                "label": label,
                "x_um": x_um,
                "y_um": y_um,
                "tile_name": tile_name,
                "coord_match_dist_um": dist,
            })
            all_rows.append(feats)
            dists.append(dist)
            matched_rows += 1

        slide_summary = {
            "slide_id": slide_id,
            "zip_path": str(zip_path),
            "n_labeled": len(slide_df),
            "n_matched": matched_rows,
            "n_missing": missing_rows,
            "mean_match_dist_um": float(np.mean(dists)) if len(dists) else np.nan,
            "max_match_dist_um": float(np.max(dists)) if len(dists) else np.nan,
        }
        slide_summaries.append(slide_summary)

        print(f"Matched rows: {matched_rows}")
        print(f"Missing rows: {missing_rows}")
        if len(dists):
            print(f"Mean coord match dist (um): {np.mean(dists):.4f}")
            print(f"Max coord match dist (um): {np.max(dists):.4f}")

# -------------------------
# Save outputs
# -------------------------
feat_df = pd.DataFrame(all_rows)
missing_df = pd.DataFrame(
    all_missing,
    columns=["slide_id", "tile_idx", "x_um", "y_um", "reason"]
)
summary_df = pd.DataFrame(slide_summaries)

feat_df.to_csv(OUT_CSV, index=False)
missing_df.to_csv("all_tile_features_missing.csv", index=False)
summary_df.to_csv("all_tile_features_summary.csv", index=False)

print(f"\nSaved feature table to: {OUT_CSV}")
print(f"Saved missing rows to: all_tile_features_missing.csv")
print(f"Saved per-slide summary to: all_tile_features_summary.csv")

print("\nFinal label counts in matched feature table:")
print(feat_df["label"].value_counts())

print("\nPer-slide summary:")
print(summary_df)

print("\nHead of feature table:")
print(feat_df.head())

# %%
import matplotlib.pyplot as plt
import pandas as pd

feat_df = pd.read_csv("all_tile_features.csv")

for col in [
    "mean_intensity", "std_intensity", "tissue_frac",
    "sat_mean", "edge_mean", "grad_x", "grad_y", "grad_ratio"
]:
    plt.figure(figsize=(5, 3.5))
    for label in ["bad_line", "good_tissue"]:
        vals = feat_df.loc[feat_df["label"] == label, col].dropna()
        plt.hist(vals, bins=40, alpha=0.5, label=label, density=True)
    plt.title(col)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# Optional: per-slide medians for quick generalizability check
summary = (
    feat_df.groupby(["slide_id", "label"])[
        ["mean_intensity", "std_intensity", "tissue_frac", "sat_mean", "edge_mean", "grad_x", "grad_y", "grad_ratio"]
    ]
    .median()
    .reset_index()
)
summary
# %%

# set up a test of features to predict the lines
# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("all_tile_features.csv")

# binary labels
df = df[df["label"].isin(["bad_line", "good_tissue"])].copy()
df["y"] = (df["label"] == "bad_line").astype(int)

# -------------------------
# Features to use
# -------------------------
FEATURES = [
    "sat_mean",
    "edge_mean",
    "grad_x",
    "grad_y",
    "grad_ratio",
    "mean_intensity",
    "std_intensity",
    "tissue_frac",
]

X = df[FEATURES].values
y = df["y"].values
groups = df["slide_id"].values

print("N samples:", len(df))
print("Slides:", df["slide_id"].nunique())

# -------------------------
# Model (scaled logistic regression)
# -------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# -------------------------
# Grouped CV
# -------------------------
gkf = GroupKFold(n_splits=5)

aucs = []
accs = []

all_preds = np.zeros_like(y, dtype=float)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\nFold {fold+1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    print(f"AUC: {auc:.3f}")
    print(f"ACC: {acc:.3f}")

    aucs.append(auc)
    accs.append(acc)

    all_preds[test_idx] = probs

# -------------------------
# Overall performance
# -------------------------
print("\n=== Overall ===")
print(f"Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"Mean ACC: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

overall_auc = roc_auc_score(y, all_preds)
print(f"OOF AUC: {overall_auc:.3f}")

# -------------------------
# Coefficients (interpretation)
# -------------------------
# retrain on all data for interpretation
model.fit(X, y)

clf = model.named_steps["clf"]
coefs = clf.coef_[0]

print("\nFeature coefficients (higher = more 'bad_line'):\n")

for f, c in sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1])):
    print(f"{f:20s} {c: .3f}")
# %%

df["pred_prob"] = all_preds

# worst false negatives
df[(df["y"]==1)].sort_values("pred_prob").head(20)

# worst false positives
df[(df["y"]==0)].sort_values("pred_prob", ascending=False).head(20)


# %%
# %%
import joblib
from pathlib import Path

MODEL_PATH = Path("/projects/marralab/rcorbett_prj/LUSC/tile_filter_model.joblib")

joblib.dump({
    "model": model,
    "features": FEATURES
}, MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH.resolve()}")
