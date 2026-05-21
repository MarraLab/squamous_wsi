#!/usr/bin/env python

from pathlib import Path
import io
import os
import zipfile
import re
import joblib
import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel

from wsi_recurrence.tile_filter.features import compute_features
from wsi_recurrence.tile_filter.zip_tiles import (
    build_zip_coord_index,
    parse_tile_coords,
)
from wsi_recurrence.env import load_local_env


load_local_env()


# -------------------------
# EDIT THESE
# -------------------------
PROJECT_DIR = Path(os.environ.get("WSI_LUSC_ROOT", "data/lusc"))
MODEL_PATH = PROJECT_DIR / "tile_filter_model.joblib"
CACHE_DIR = Path(os.environ.get("WSI_IMAGE_CACHE", os.environ.get("WSI_CACHE_ROOT", ".cache/image_cache") + "/lusc"))
SLIDE_ID = "R_013"   # change for testing
OUT_CSV = Path(f"{SLIDE_ID}_tile_predictions.csv")

THRESHOLD = 0.3
COORD_TOL = 2.0

# -------------------------
# Feature function (must match training!)
# -------------------------


# -------------------------
# Load model
# -------------------------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

print("Loaded model")
print("Features:", FEATURES)


# -------------------------
# Find zip
# -------------------------
zips = list(CACHE_DIR.glob(f"{SLIDE_ID}.*.zip"))
if not zips:
    raise FileNotFoundError(f"No zip found for {SLIDE_ID}")
zip_path = zips[0]

print(f"Using zip: {zip_path}")


# -------------------------
# Process slide
# -------------------------
rows = []

with zipfile.ZipFile(zip_path, "r") as zf:
    zip_df = build_zip_coord_index(zf.namelist(), x_col="x_um", y_col="y_um")

    print(f"Tiles in zip: {len(zip_df)}")

    for _, row in zip_df.iterrows():
        tile_name = row["tile_name"]

        img_bytes = zf.read(tile_name)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        feats = compute_features(img)

        rows.append({
            "slide_id": SLIDE_ID,
            "tile_name": tile_name,
            **feats
        })

feat_df = pd.DataFrame(rows)

print("Computed features:", feat_df.shape)

# -------------------------
# Predict
# -------------------------
X = feat_df[FEATURES].values
probs = model.predict_proba(X)[:, 1]

# Apply threshold
feat_df["bad_prob"] = probs
feat_df["keep"] = probs < THRESHOLD

print("\nSummary:")
print(feat_df["keep"].value_counts())

# -------------------------
# Save
# -------------------------
feat_df.to_csv(OUT_CSV, index=False)
print(f"Saved predictions to: {OUT_CSV}")

# -------------------------
# QC visualization
# -------------------------
import matplotlib.pyplot as plt
import openslide

print("\nGenerating QC overlay...")

# Load slide thumbnail
ndpi_path = PROJECT_DIR / f"{SLIDE_ID}.ndpi"

slide = openslide.OpenSlide(str(ndpi_path))
full_w, full_h = slide.dimensions

thumb_w = 1600
thumb_h = int(round(full_h * (thumb_w / full_w)))
thumb = np.array(slide.get_thumbnail((thumb_w, thumb_h)))

# Convert coords from tile names → microns → pixels → thumbnail coords
coords = np.array([parse_tile_coords(t) for t in feat_df["tile_name"]])
x_um = coords[:, 0]
y_um = coords[:, 1]

# microns → pixels
PIXEL_WIDTH_UM = 0.2298
PIXEL_HEIGHT_UM = 0.2299

x_px = x_um / PIXEL_WIDTH_UM
y_px = y_um / PIXEL_HEIGHT_UM

# pixels → thumbnail coords
x_thumb = x_px * (thumb_w / full_w)
y_thumb = y_px * (thumb_h / full_h)

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10, 8))
plt.imshow(thumb)

# plot kept vs removed
keep_mask = feat_df["keep"].values

plt.scatter(
    x_thumb[keep_mask],
    y_thumb[keep_mask],
    s=4,
    c="lime",
    alpha=0.5,
    label=f"kept ({keep_mask.sum()})"
)

plt.scatter(
    x_thumb[~keep_mask],
    y_thumb[~keep_mask],
    s=6,
    c="red",
    alpha=0.8,
    label=f"removed ({(~keep_mask).sum()})"
)

plt.title(f"{SLIDE_ID} – {THRESHOLD:.2f} tile filter QC")
plt.legend(loc="lower right")
plt.axis("off")

OUT_PNG = Path(f"{SLIDE_ID}_{THRESHOLD:.2f}_filter_qc.png")
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved QC image: {OUT_PNG}")
