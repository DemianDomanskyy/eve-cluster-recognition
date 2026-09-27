"""Recovering a point cloud from a screenshot of a scatter plot.

Project Discovery is played inside the client, so the plate you want to solve is
usually pixels on screen rather than a data file.  This module thresholds the
image, finds the connected blobs, and returns one coordinate per plotted marker -
splitting blobs that are clearly several overlapping markers fused together.

Nothing here is machine learning; it is deliberately simple and inspectable, and
its output feeds the same `Plate` type as every other input path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .types import Plate


def _otsu_threshold(gray: np.ndarray) -> float:
    """Classic Otsu split of a greyscale histogram."""
    hist, edges = np.histogram(gray.ravel(), bins=256, range=(0.0, 255.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0
    omega = np.cumsum(hist) / total
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(hist * centers) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.argmax(sigma_b))])


def load_image_gray(path: str | Path, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Load an image as a float greyscale array, optionally cropped to the plot area."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if crop is not None:
        img = img.crop(crop)
    return np.asarray(img.convert("L"), dtype=np.float64)


def points_from_image(
    path: str | Path,
    crop: tuple[int, int, int, int] | None = None,
    min_area: int = 2,
    max_area_ratio: float = 0.02,
    max_aspect: float = 6.0,
    split_large_blobs: bool = True,
    threshold: float | None = None,
    invert: bool | None = None,
    plate_id: str | None = None,
) -> Plate:
    """Extract marker positions from a scatter-plot image.

    Args:
        crop: (left, top, right, bottom) pixel box of the plot area, excluding axes.
        min_area: ignore blobs smaller than this many pixels (anti-aliasing specks).
        max_area_ratio: ignore blobs larger than this share of the image (axes, text).
        max_aspect: ignore very elongated blobs, which are gridlines and axis rules.
        split_large_blobs: split fused markers into several points by area.
        threshold: greyscale cutoff; Otsu's method is used when omitted.
        invert: force marker polarity; auto-detected from background brightness.
    """
    gray = load_image_gray(path, crop)
    if gray.size == 0:
        raise ValueError("image (or crop region) is empty")

    thresh = _otsu_threshold(gray) if threshold is None else float(threshold)
    dark_on_light = float(np.mean(gray)) > thresh if invert is None else bool(invert)
    mask = gray < thresh if dark_on_light else gray > thresh

    labelled, n_blobs = ndimage.label(mask)
    if n_blobs == 0:
        return Plate(points=np.zeros((0, 2)), plate_id=plate_id, meta={"source": str(path)})

    objects = ndimage.find_objects(labelled)
    areas = ndimage.sum(mask, labelled, index=np.arange(1, n_blobs + 1))
    max_area = max_area_ratio * gray.size

    kept: list[tuple[int, float, tuple[slice, slice]]] = []
    for idx, (slices, area) in enumerate(zip(objects, areas), start=1):
        if area < min_area or area > max_area:
            continue
        h = slices[0].stop - slices[0].start
        w = slices[1].stop - slices[1].start
        aspect = max(h, w) / max(1, min(h, w))
        if aspect > max_aspect:
            continue
        kept.append((idx, float(area), slices))

    if not kept:
        return Plate(points=np.zeros((0, 2)), plate_id=plate_id, meta={"source": str(path)})

    median_area = float(np.median([a for _, a, _ in kept]))
    coords: list[np.ndarray] = []

    for idx, area, _slices in kept:
        pixels = np.argwhere(labelled == idx).astype(np.float64)  # (row, col)
        n_markers = 1
        if split_large_blobs and median_area > 0:
            n_markers = int(round(area / median_area))
            n_markers = max(1, min(n_markers, 40))
        if n_markers == 1 or len(pixels) < n_markers * 2:
            coords.append(pixels.mean(axis=0))
        else:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_markers, n_init=3, random_state=0).fit(pixels)
            coords.extend(km.cluster_centers_)

    arr = np.asarray(coords, dtype=np.float64)
    height, width = gray.shape
    # Image rows grow downward; plots grow upward, so flip y.  Dividing by
    # (size - 1) inverts the pixel mapping exactly rather than off by one pixel.
    points = np.stack(
        [arr[:, 1] / max(width - 1, 1), 1.0 - arr[:, 0] / max(height - 1, 1)], axis=1
    )

    return Plate(
        points=points,
        plate_id=plate_id,
        meta={
            "source": str(path),
            "image_size": [int(width), int(height)],
            "threshold": round(thresh, 2),
            "dark_on_light": bool(dark_on_light),
            "blobs_found": int(n_blobs),
            "blobs_kept": len(kept),
            "markers": int(len(points)),
        },
    )


def render_plate_image(
    plate: Plate,
    path: str | Path,
    size: int = 512,
    radius: int = 2,
    dark_on_light: bool = True,
) -> Path:
    """Draw a plate as a plain scatter image.  Used by the tests and for round-tripping."""
    from PIL import Image, ImageDraw

    bg, fg = ((255, 255, 255), (20, 20, 20)) if dark_on_light else ((10, 10, 20), (240, 240, 240))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    pts = plate.normalized().points
    # Inset by the marker radius: a marker drawn hard against the border would be
    # clipped, and its recovered centroid would be pulled inward.
    margin = radius + 1
    extent = max(1, size - 1 - 2 * margin)
    for x, y in pts:
        cx = margin + float(np.clip(x, 0, 1)) * extent
        cy = margin + (1.0 - float(np.clip(y, 0, 1))) * extent
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
