#!/usr/bin/env python
from pathlib import Path
import argparse
import shutil
import sys

import h5py
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from wsi_recurrence.tile_filter.h5_filter import filter_h5, load_keep_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir = Path(args.mask_dir)

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Mask dir: {mask_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} h5 files")

    for h5_path in tqdm(h5_files, desc=f"Filtering {input_dir.name}", unit="slide"):
        slide_id = h5_path.stem
        mask_csv = mask_dir / f"{slide_id}_keep_mask.csv"
        out_h5 = output_dir / h5_path.name

        if not mask_csv.exists():
            print(f"{slide_id}: no mask found, copying unfiltered")
            shutil.copy2(h5_path, out_h5)
            continue

        keep_map = load_keep_mask(mask_csv)
        n_total, n_kept = filter_h5(h5_path, out_h5, keep_map)
        print(f"{slide_id}: kept {n_kept}/{n_total}")

    # copy non-h5 files if present
    for extra in input_dir.iterdir():
        if extra.suffix != ".h5" and extra.is_file():
            shutil.copy2(extra, output_dir / extra.name)


if __name__ == "__main__":
    main()
