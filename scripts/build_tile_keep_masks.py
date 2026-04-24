#!/usr/bin/env python
from pathlib import Path
import io
import zipfile
import re
import joblib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel

from wsi_recurrence.tile_filter.features import compute_features
from wsi_recurrence.tile_filter.zip_tiles import build_zip_coord_index

MODEL_PATH = Path("/projects/marralab/rcorbett_prj/LUSC/tile_filter_model.joblib")
CACHE_DIR = Path("/tmp/image_cache")
OUT_DIR = Path("/projects/marralab/rcorbett_prj/LUSC/tile_keep_masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.30
ROUND_DECIMALS = 3
MAX_WORKERS = 16

def process_one_zip(zip_path_str: str, model_path_str: str, out_dir_str: str,
                    threshold: float, round_decimals: int) -> dict:
    zip_path = Path(zip_path_str)
    out_dir = Path(out_dir_str)

    slide_id = zip_path.name.split(".")[0]
    out_csv = out_dir / f"{slide_id}_keep_mask.csv"

    if out_csv.exists():
        return {
            "slide_id": slide_id,
            "status": "skipped_exists",
            "out_csv": str(out_csv),
        }

    bundle = joblib.load(model_path_str)
    model = bundle["model"]
    features = bundle["features"]

    with zipfile.ZipFile(zip_path, "r") as zf:
        coord_df = build_zip_coord_index(zf.namelist(), x_col="x_um", y_col="y_um")
        rows = []

        for _, row in coord_df.iterrows():
            tile_name = row["tile_name"]
            img = Image.open(io.BytesIO(zf.read(tile_name))).convert("RGB")
            feats = compute_features(img)
            rows.append({
                "tile_name": tile_name,
                "x_um": row["x_um"],
                "y_um": row["y_um"],
                **feats
            })

    feat_df = pd.DataFrame(rows)
    probs = model.predict_proba(feat_df[features].values)[:, 1]
    feat_df["bad_prob"] = probs
    feat_df["keep"] = probs < threshold
    feat_df["x_key"] = feat_df["x_um"].round(round_decimals)
    feat_df["y_key"] = feat_df["y_um"].round(round_decimals)

    feat_df.to_csv(out_csv, index=False)

    return {
        "slide_id": slide_id,
        "status": "ok",
        "out_csv": str(out_csv),
        "n_tiles": len(feat_df),
        "n_kept": int(feat_df["keep"].sum()),
        "n_removed": int((~feat_df["keep"]).sum()),
    }


def main():
    zip_paths = sorted(CACHE_DIR.glob("*.zip"))
    print(f"Found {len(zip_paths)} zip files")
    print(f"Using {MAX_WORKERS} workers")

    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for zip_path in zip_paths:
            futures.append(
                ex.submit(
                    process_one_zip,
                    str(zip_path),
                    str(MODEL_PATH),
                    str(OUT_DIR),
                    THRESHOLD,
                    ROUND_DECIMALS,
                )
            )

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
                results.append(res)
                if res["status"] == "ok":
                    print(
                        f"[{i}/{len(futures)}] {res['slide_id']}: "
                        f"kept {res['n_kept']}/{res['n_tiles']}"
                    )
                else:
                    print(f"[{i}/{len(futures)}] {res['slide_id']}: {res['status']}")
            except Exception as e:
                print(f"[{i}/{len(futures)}] ERROR: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "keep_mask_build_summary.csv", index=False)
    print(f"\nSaved summary to: {OUT_DIR / 'keep_mask_build_summary.csv'}")


if __name__ == "__main__":
    main()
