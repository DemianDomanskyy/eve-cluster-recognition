"""Rendering a solved plate to PNG, and the same numbers as a text table.

Color choices here are validated, not eyeballed.  This is a scatter, so every
pair of hues is on screen at once and has to be separable under simulated
colorblindness: the palettes below are the largest sets that clear that bar
(4 hues on the light surface, 3 on the dark one).  A plate with more clusters
than that does not cycle hues - it folds to one neutral ink and lets the outline
plus the direct number carry identity, which is unambiguous anyway because
clusters occupy different places on the plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .polygons import Polygon
from .types import NOISE, Plate, Solution

# Validated categorical slots (OKLab CVD separation, all-pairs, in both modes).
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70"]

THEME_LIGHT = {
    "surface": "#fcfcfb",
    "text_primary": "#1a1a19",
    "text_secondary": "#5c5b52",
    "muted": "#a8a79c",
    "grid": "#e8e7e1",
    "folded": "#5c5b52",
}
THEME_DARK = {
    "surface": "#1a1a19",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#6b6a61",
    "grid": "#2e2e2c",
    "folded": "#c3c2b7",
}


def cluster_table(plate: Plate, solution: Solution) -> list[dict[str, object]]:
    """The chart's numbers as rows, so the figure is never the only way to read it."""
    labels = solution.labels
    rows: list[dict[str, object]] = []
    for i, lab in enumerate(sorted(set(labels.tolist()) - {NOISE})):
        pts = plate.points[labels == lab]
        rows.append(
            {
                "cluster": i + 1,
                "points": int(len(pts)),
                "share": round(float(len(pts) / max(len(labels), 1)), 4),
                "center_x": round(float(pts[:, 0].mean()), 4),
                "center_y": round(float(pts[:, 1].mean()), 4),
                "spread": round(float(np.mean(np.std(pts, axis=0))), 4),
            }
        )
    n_noise = int(np.count_nonzero(labels == NOISE))
    rows.append(
        {
            "cluster": "unclustered",
            "points": n_noise,
            "share": round(float(n_noise / max(len(labels), 1)), 4),
            "center_x": "",
            "center_y": "",
            "spread": "",
        }
    )
    return rows


def format_cluster_table(rows: list[dict[str, object]]) -> str:
    """Render `cluster_table` rows as fixed-width text for the terminal."""
    headers = ["cluster", "points", "share", "center_x", "center_y", "spread"]
    widths = {
        h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) if rows else len(h)
        for h in headers
    }
    lines = ["  ".join(h.ljust(widths[h]) for h in headers)]
    lines.append("  ".join("-" * widths[h] for h in headers))
    for r in rows:
        lines.append("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))
    return "\n".join(lines)


def plot_solution(
    plate: Plate,
    solution: Solution,
    path: str | Path,
    title: str | None = None,
    dark: bool = False,
    dpi: int = 160,
    show_polygons: bool = True,
) -> Path:
    """Save a figure of the plate with its clusters outlined and numbered."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = THEME_DARK if dark else THEME_LIGHT
    palette = PALETTE_DARK if dark else PALETTE_LIGHT

    labels = solution.labels
    cluster_ids = sorted(set(labels.tolist()) - {NOISE})
    folded = len(cluster_ids) > len(palette)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    noise_mask = labels == NOISE
    if noise_mask.any():
        ax.scatter(
            plate.points[noise_mask, 0],
            plate.points[noise_mask, 1],
            s=5,
            c=theme["muted"],
            alpha=0.55,
            linewidths=0,
            label="unclustered",
            zorder=1,
        )

    polygons: list[Polygon] = solution.polygons or []
    for i, lab in enumerate(cluster_ids):
        color = theme["folded"] if folded else palette[i]
        mask = labels == lab
        pts = plate.points[mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=9,
            c=color,
            alpha=0.9,
            linewidths=0,
            label=f"cluster {i + 1} ({int(mask.sum())} pts)",
            zorder=2,
        )
        if show_polygons and i < len(polygons) and polygons[i]:
            ring = np.asarray(polygons[i] + [polygons[i][0]], dtype=float)
            ax.plot(ring[:, 0], ring[:, 1], color=color, lw=2.0, alpha=0.95, zorder=3)
        # Direct label: identity never rests on hue alone.  Anchored to the
        # medoid, not the centroid - a crescent's centroid sits in the empty
        # space inside its bend, which would float the number off its cluster.
        centroid = pts.mean(axis=0)
        cx, cy = pts[np.argmin(np.linalg.norm(pts - centroid, axis=1))]
        ax.annotate(
            str(i + 1),
            (cx, cy),
            color=theme["text_primary"],
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
            bbox={"boxstyle": "circle,pad=0.25", "fc": theme["surface"], "ec": color, "lw": 1.5},
        )

    if title is None:
        title = f"{len(cluster_ids)} cluster(s) · {solution.algorithm} · {solution.source}"
    ax.set_title(title, color=theme["text_primary"], fontsize=12, pad=12, loc="left")

    ax.grid(True, color=theme["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["grid"])
    ax.tick_params(colors=theme["text_secondary"], labelsize=9)

    if len(cluster_ids) >= 1:
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.06),
            ncols=2,
            frameon=False,
            fontsize=9,
            labelcolor=theme["text_secondary"],
        )
        if legend is not None:
            legend.set_zorder(5)

    if folded:
        ax.text(
            1.0,
            -0.06,
            f"{len(cluster_ids)} clusters: numbered, not color-coded",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=theme["text_secondary"],
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)
    return path
