# %%
from pathlib import Path
import csv

import h5py
import numpy as np
import openslide
from PIL import Image

import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

# -------------------------
# Edit these
# -------------------------
SLIDE_IDS = [
    "R_106",
    "R_080",
    "R_096",
    "R_146",
    "R_008",
    "Sq_056_T",
    "GSC2-4",
    "GSC2-103_TNA",
    "GSC2-25",
    "P0059",
    "P0071",
    "R_021",
    "R_022",
    "R_059",
    "Sq_008_T",
    
]

WSI_DIR = Path("/projects/marralab/rcorbett_prj/LUSC")
H5_DIR = Path("/projects/marralab/rcorbett_prj/LUSC/stamp_preprocess/dino-bloom/wsi/dino-bloom-e8eb3d28")
LABEL_DIR = Path("./tile_labels")
LABEL_DIR.mkdir(exist_ok=True)

THUMB_MAX_W = 1600
PIXEL_WIDTH_UM = 0.2298
PIXEL_HEIGHT_UM = 0.2299
POINT_SIZE = 5

LABEL_TO_COLOR = {
    "unlabeled": "lime",
    "bad_line": "red",
    "good_tissue": "blue",
}

# -------------------------
# Helpers
# -------------------------
def ndpi_path_for(slide_id: str) -> Path:
    return WSI_DIR / f"{slide_id}.ndpi"

def h5_path_for(slide_id: str) -> Path:
    return H5_DIR / f"{slide_id}.h5"

def csv_path_for(slide_id: str) -> Path:
    return LABEL_DIR / f"{slide_id}_tile_labels.csv"

def load_thumbnail_from_wsi(ndpi_path: Path, thumb_max_w: int = 1600):
    slide = openslide.OpenSlide(str(ndpi_path))
    full_w, full_h = slide.dimensions
    thumb_w = thumb_max_w
    thumb_h = int(round(full_h * (thumb_w / full_w)))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    return slide, np.array(thumb)

def load_h5_coords_um(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        coords_um = f["coords"][:]
    return coords_um

def map_um_to_thumbnail(coords_um: np.ndarray, thumb_shape, full_w_px: int, full_h_px: int):
    thumb_h, thumb_w, _ = thumb_shape
    x_full = coords_um[:, 0] / PIXEL_WIDTH_UM
    y_full = coords_um[:, 1] / PIXEL_HEIGHT_UM
    x_thumb = x_full * (thumb_w / full_w_px)
    y_thumb = y_full * (thumb_h / full_h_px)
    return x_thumb, y_thumb

def load_existing_labels(csv_path: Path, n_tiles: int):
    labels = np.array(["unlabeled"] * n_tiles, dtype=object)
    if not csv_path.exists():
        return labels

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_idx = int(row["tile_idx"])
            label = row["label"]
            if 0 <= tile_idx < n_tiles:
                labels[tile_idx] = label
    return labels

def save_labels_csv(out_csv: Path, slide_id: str, labels, coords_um, x_thumb, y_thumb):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "tile_idx", "label", "x_um", "y_um", "x_thumb", "y_thumb"])
        for i, label in enumerate(labels):
            if label == "unlabeled":
                continue
            writer.writerow([
                slide_id,
                i,
                label,
                float(coords_um[i, 0]),
                float(coords_um[i, 1]),
                float(x_thumb[i]),
                float(y_thumb[i]),
            ])

def colors_from_labels(labels):
    return [LABEL_TO_COLOR.get(x, "lime") for x in labels]

# -------------------------
# State
# -------------------------
current_slide_idx = 0
current_slide_id = None
slide = None
thumb_np = None
coords_um = None
x_thumb = None
y_thumb = None
labels = None

# -------------------------
# UI
# -------------------------
slide_dropdown = widgets.Dropdown(
    options=SLIDE_IDS,
    value=SLIDE_IDS[0],
    description="Slide:",
    layout=widgets.Layout(width="250px")
)

prev_button = widgets.Button(description="Prev")
next_button = widgets.Button(description="Next")

label_dropdown = widgets.Dropdown(
    options=["bad_line", "good_tissue", "unlabeled"],
    value="bad_line",
    description="Label:",
    layout=widgets.Layout(width="220px")
)

apply_button = widgets.Button(description="Apply label to selected")
save_button = widgets.Button(description="Save CSV")
inspect_button = widgets.Button(description="Show selection size")
status = widgets.Output(layout={"border": "1px solid #ccc"})

fig = go.FigureWidget()
scatter = None

# -------------------------
# Core functions
# -------------------------
def counts_for_labels(lbls):
    vals, counts = np.unique(lbls, return_counts=True)
    d = dict(zip(vals, counts))
    return {
        "bad_line": int(d.get("bad_line", 0)),
        "good_tissue": int(d.get("good_tissue", 0)),
        "unlabeled": int(d.get("unlabeled", 0)),
    }

def update_title():
    global current_slide_id, labels
    c = counts_for_labels(labels)
    fig.update_layout(
        title=(
            f"{current_slide_id}: lasso-select tiles | "
            f"bad_line={c['bad_line']}  "
            f"good_tissue={c['good_tissue']}  "
            f"unlabeled={c['unlabeled']}"
        )
    )

def get_selected_indices():
    selected = fig.data[0].selectedpoints
    if selected is None:
        return []
    return list(selected)

def update_plot_colors():
    global labels
    with fig.batch_update():
        fig.data[0].marker.color = colors_from_labels(labels)

def build_or_replace_figure():
    global fig, scatter, thumb_np, x_thumb, y_thumb, labels

    fig.data = ()
    fig.layout.images = ()

    pil_img = Image.fromarray(thumb_np)
    fig.add_layout_image(
        dict(
            source=pil_img,
            x=0,
            y=0,
            sizex=thumb_np.shape[1],
            sizey=thumb_np.shape[0],
            xref="x",
            yref="y",
            layer="below",
        )
    )

    fig.add_scattergl(
        x=x_thumb,
        y=y_thumb,
        mode="markers",
        marker=dict(
            size=POINT_SIZE,
            color=colors_from_labels(labels),
            opacity=0.7,
        ),
        selected=dict(marker=dict(color="yellow", size=POINT_SIZE + 2)),
        unselected=dict(marker=dict(opacity=0.7)),
        customdata=np.arange(len(coords_um)),
        hovertemplate="tile_idx=%{customdata}<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
    )

    fig.update_xaxes(range=[0, thumb_np.shape[1]], visible=False)
    fig.update_yaxes(range=[thumb_np.shape[0], 0], visible=False, scaleanchor="x")
    fig.update_layout(
        width=1000,
        height=int(1000 * thumb_np.shape[0] / thumb_np.shape[1]),
        dragmode="lasso",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    update_title()

def load_slide(slide_id: str):
    global current_slide_id, current_slide_idx
    global slide, thumb_np, coords_um, x_thumb, y_thumb, labels

    ndpi_path = ndpi_path_for(slide_id)
    h5_path = h5_path_for(slide_id)
    out_csv = csv_path_for(slide_id)

    if not ndpi_path.exists():
        raise FileNotFoundError(f"Missing NDPI: {ndpi_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing H5: {h5_path}")

    slide, thumb_np = load_thumbnail_from_wsi(ndpi_path, THUMB_MAX_W)
    full_w, full_h = slide.dimensions
    coords_um = load_h5_coords_um(h5_path)
    x_thumb, y_thumb = map_um_to_thumbnail(coords_um, thumb_np.shape, full_w, full_h)
    labels = load_existing_labels(out_csv, len(coords_um))

    current_slide_id = slide_id
    current_slide_idx = SLIDE_IDS.index(slide_id)

    build_or_replace_figure()

    with status:
        status.clear_output()
        print(f"Loaded {slide_id}")
        print(f"Tiles: {len(coords_um)}")
        print(f"CSV: {out_csv}")

def save_current_labels():
    out_csv = csv_path_for(current_slide_id)
    save_labels_csv(out_csv, current_slide_id, labels, coords_um, x_thumb, y_thumb)
    n_saved = int(np.sum(labels != "unlabeled"))
    with status:
        status.clear_output()
        print(f"Saved labels to {out_csv}")
        print(f"Number of labeled tiles saved: {n_saved}")

# -------------------------
# Callbacks
# -------------------------
def on_apply_clicked(b):
    global labels
    selected = get_selected_indices()

    with status:
        status.clear_output()
        print(f"Apply pressed. Current label = {label_dropdown.value}")
        print(f"Selected indices = {len(selected)}")

    if len(selected) == 0:
        with status:
            print("No tiles selected")
        return

    for idx in selected:
        labels[idx] = label_dropdown.value

    update_plot_colors()
    update_title()

    with status:
        print(f"Applied label '{label_dropdown.value}' to {len(selected)} tiles on {current_slide_id}")

def on_save_clicked(b):
    save_current_labels()

def on_inspect_clicked(b):
    selected = get_selected_indices()
    with status:
        status.clear_output()
        print(f"Selected {len(selected)} tiles on {current_slide_id}")

def on_prev_clicked(b):
    global current_slide_idx
    save_current_labels()
    new_idx = max(0, current_slide_idx - 1)
    slide_dropdown.value = SLIDE_IDS[new_idx]

def on_next_clicked(b):
    global current_slide_idx
    save_current_labels()
    new_idx = min(len(SLIDE_IDS) - 1, current_slide_idx + 1)
    slide_dropdown.value = SLIDE_IDS[new_idx]

def on_slide_changed(change):
    if change["name"] == "value" and change["new"] is not None:
        load_slide(change["new"])

apply_button.on_click(on_apply_clicked)
save_button.on_click(on_save_clicked)
inspect_button.on_click(on_inspect_clicked)
prev_button.on_click(on_prev_clicked)
next_button.on_click(on_next_clicked)
slide_dropdown.observe(on_slide_changed, names="value")

# -------------------------
# Display
# -------------------------
display(widgets.HBox([slide_dropdown, prev_button, next_button]))
display(widgets.HBox([label_dropdown, apply_button, save_button, inspect_button]))
display(status)

load_slide(SLIDE_IDS[0])
fig
# %%