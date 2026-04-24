from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd


ROUND_DECIMALS = 3


def load_keep_mask(mask_csv: Path) -> Dict[Tuple[float, float], bool]:
    df = pd.read_csv(mask_csv)
    keep_map = {
        (round(float(x), ROUND_DECIMALS), round(float(y), ROUND_DECIMALS)): bool(k)
        for x, y, k in zip(df["x_um"], df["y_um"], df["keep"])
    }
    return keep_map


def filter_h5(in_h5: Path, out_h5: Path, keep_map: dict) -> Tuple[int, int]:
    with h5py.File(in_h5, "r") as fin:
        coords = fin["coords"][:]
        feats = fin["feats"][:]

        keys = [
            (round(float(x), ROUND_DECIMALS), round(float(y), ROUND_DECIMALS))
            for x, y in coords
        ]

        keep = np.array([keep_map.get(k, True) for k in keys], dtype=bool)

        if keep.sum() == 0:
            raise RuntimeError(f"All tiles removed for {in_h5.name}")

        out_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_h5, "w") as fout:
            for k, v in fin.attrs.items():
                fout.attrs[k] = v

            for ds_name in fin.keys():
                arr = fin[ds_name][:]
                if arr.shape[0] == len(keep):
                    fout.create_dataset(ds_name, data=arr[keep], compression="gzip")
                else:
                    fout.create_dataset(ds_name, data=arr, compression="gzip")

    return int(len(keep)), int(keep.sum())

