from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


TILE_FILENAME_RE = re.compile(r"tile_\(([^,]+), ([^)]+)\)\.jpg$")


def parse_tile_coords(tile_name: str) -> Tuple[float, float]:
    """
    Extract (x_um, y_um) from a tile filename like: tile_(x, y).jpg

    This intentionally mirrors prior script behavior: if the name does not match
    the expected pattern, it will raise due to attribute access on None.
    """
    m = TILE_FILENAME_RE.match(tile_name)
    return float(m.group(1)), float(m.group(2))


def try_parse_tile_coords(tile_name: str) -> Optional[Tuple[float, float]]:
    m = TILE_FILENAME_RE.match(tile_name)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def build_zip_coord_index(
    zip_names: Iterable[str],
    *,
    x_col: str = "x_um_zip",
    y_col: str = "y_um_zip",
) -> pd.DataFrame:
    rows = []
    for name in zip_names:
        coords = try_parse_tile_coords(name)
        if coords is None:
            continue
        x_um, y_um = coords
        rows.append((name, x_um, y_um))
    return pd.DataFrame(rows, columns=["tile_name", x_col, y_col])


def match_tile_name(
    x_um: float,
    y_um: float,
    zip_df: pd.DataFrame,
    *,
    tol: float = 2.0,
    x_col: str = "x_um_zip",
    y_col: str = "y_um_zip",
) -> Tuple[Optional[str], Optional[float]]:
    if len(zip_df) == 0:
        return None, None

    dx = np.abs(zip_df[x_col].values - x_um)
    dy = np.abs(zip_df[y_col].values - y_um)
    d = np.sqrt(dx * dx + dy * dy)
    i = int(np.argmin(d))
    if d[i] <= tol:
        return str(zip_df.iloc[i]["tile_name"]), float(d[i])
    return None, None

