"""Reading and writing plates: npz, csv, json, and screenshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import NOISE, Plate

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def load_plate(path: str | Path, **image_kwargs) -> Plate:
    """Load a plate from any supported file type, dispatching on the extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        from .vision import points_from_image

        return points_from_image(path, plate_id=path.stem, **image_kwargs)

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            labels = data["labels"] if "labels" in data else None
            return Plate(points=data["points"], labels=labels, plate_id=path.stem)

    if suffix == ".npy":
        return Plate(points=np.load(path, allow_pickle=False), plate_id=path.stem)

    if suffix in {".csv", ".tsv", ".txt"}:
        return _load_delimited(path)

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Plate(
            points=np.asarray(payload["points"], dtype=np.float64),
            labels=None if payload.get("labels") is None else np.asarray(payload["labels"]),
            plate_id=payload.get("plate_id", path.stem),
            meta=payload.get("meta", {}),
        )

    raise ValueError(f"unsupported plate format: {path.suffix!r}")


def _load_delimited(path: Path) -> Plate:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
    has_header = any(c.isalpha() for c in first.replace("e", "").replace("E", ""))
    arr = np.genfromtxt(
        path, delimiter=delimiter, skip_header=1 if has_header else 0, dtype=np.float64
    )
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise ValueError(f"{path} needs at least two columns (x, y)")
    labels = None
    if arr.shape[1] >= 3:
        labels = np.nan_to_num(arr[:, 2], nan=NOISE).astype(np.int64)
    return Plate(points=arr[:, :2], labels=labels, plate_id=path.stem)


def save_plate(plate: Plate, path: str | Path) -> Path:
    """Write a plate as compressed npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"points": plate.points}
    if plate.labels is not None:
        arrays["labels"] = plate.labels
    np.savez_compressed(path, **arrays)
    return path


def save_dataset(plates: list[Plate], directory: str | Path) -> Path:
    """Write a list of plates as `plate_0000.npz`, ... plus a manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, plate in enumerate(plates):
        name = f"plate_{i:04d}.npz"
        save_plate(plate, directory / name)
        manifest.append(
            {
                "file": name,
                "plate_id": plate.plate_id,
                "n_points": plate.n_points,
                "n_clusters": plate.n_true_clusters,
                "meta": plate.meta,
            }
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def load_dataset(directory: str | Path) -> list[Plate]:
    """Load every plate written by `save_dataset`, in filename order."""
    directory = Path(directory)
    files = sorted(directory.glob("plate_*.npz"))
    if not files:
        raise FileNotFoundError(f"no plate_*.npz files in {directory}")
    return [load_plate(f) for f in files]
