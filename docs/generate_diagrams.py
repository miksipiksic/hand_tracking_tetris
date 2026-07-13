"""
Generate the English documentation diagrams used by README.md.

This is a self-contained documentation asset: it does NOT import or modify any
of the project's game / ML code. It only draws two figures with matplotlib and
saves them into docs/assets/:

    - architecture.png   : real-time inference pipeline (camera -> render)
    - data_pipeline.png  : offline data & training pipeline (collect -> evaluate)

Usage:
    python docs/generate_diagrams.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---- shared style -----------------------------------------------------------
PALETTE = {
    "input":  ("#E3F2FD", "#1565C0"),   # light fill, strong border/text
    "vision": ("#E8F5E9", "#2E7D32"),
    "ml":     ("#F3E5F5", "#6A1B9A"),
    "game":   ("#FFF3E0", "#EF6C00"),
}
BOX_W, BOX_H = 2.9, 1.15
FONT = dict(fontsize=11, ha="center", va="center", weight="bold")


def _box(ax, cx, cy, title, subtitle, kind):
    fill, edge = PALETTE[kind]
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(cx, cy + 0.16, title, color=edge, **FONT)
    if subtitle:
        ax.text(cx, cy - 0.24, subtitle, color=edge, fontsize=8.5,
                ha="center", va="center")


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
        linewidth=1.8, color="#555555", zorder=1,
        shrinkA=4, shrinkB=4))


def architecture_diagram():
    """Real-time inference pipeline, laid out as a U (3 top, 3 bottom)."""
    nodes = [
        # (x, y, title, subtitle, kind)
        (0,  1, "Webcam",           "OpenCV frame capture",      "input"),
        (4,  1, "MediaPipe Hands",  "21 landmarks / hand",       "vision"),
        (8,  1, "Feature vector",   "42 x/y coords -> Scaler",   "vision"),
        (8, -1, "MLP Classifier",   "5-class gesture, 99.98%",   "ml"),
        (4, -1, "Gesture -> Command","static + dynamic gestures", "ml"),
        (0, -1, "Tetris + pygame",  "game logic & rendering",    "game"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    for n in nodes:
        _box(ax, *n)
    # top row left->right
    _arrow(ax, 0 + BOX_W / 2, 1, 4 - BOX_W / 2, 1)
    _arrow(ax, 4 + BOX_W / 2, 1, 8 - BOX_W / 2, 1)
    # down the right side
    _arrow(ax, 8, 1 - BOX_H / 2, 8, -1 + BOX_H / 2)
    # bottom row right->left
    _arrow(ax, 8 - BOX_W / 2, -1, 4 + BOX_W / 2, -1)
    _arrow(ax, 4 - BOX_W / 2, -1, 0 + BOX_W / 2, -1)

    ax.set_xlim(-2, 10)
    ax.set_ylim(-2.2, 2.2)
    ax.axis("off")
    ax.set_title("Real-Time Inference Pipeline",
                 fontsize=15, weight="bold", pad=14)
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "architecture.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def data_pipeline_diagram():
    """Offline data collection & model training pipeline (linear)."""
    stages = [
        ("Collect",  "create_csv_with_gestures.py\nlandmarks via webcam", "input"),
        ("Augment",  "process_csv.py\nflip / rotate / scale / noise",     "vision"),
        ("Clean",    "clean_csv.py\nnormalize CSV rows",                  "vision"),
        ("Merge",    "merge_csv.py\noriginal + augmented",                "vision"),
        ("Train & Evaluate", "multi_process.py\nRF / MLP / XGBoost, 5-fold CV", "ml"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.2))
    xs = [0, 3.6, 7.2, 10.8, 14.4]
    for x, (title, sub, kind) in zip(xs, stages):
        _box(ax, x, 0, title, sub, kind)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        _arrow(ax, x1 + BOX_W / 2, 0, x2 - BOX_W / 2, 0)

    ax.set_xlim(-2, 16.4)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    ax.set_title("Data & Training Pipeline",
                 fontsize=15, weight="bold", pad=12)
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "data_pipeline.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    architecture_diagram()
    data_pipeline_diagram()
    print("Done. Diagrams written to", ASSETS_DIR)
