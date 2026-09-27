"""Core data structures shared by every stage of the pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

NOISE = -1
"""Label used for points that belong to no cluster (background / debris)."""


def _as_points(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {arr.shape}")
    return arr


@dataclass
class Plate:
    """A single task: a 2D point cloud, optionally with ground-truth labels.

    The name follows the flow-cytometry vocabulary used by the Project Discovery
    COVID-19 phase, where each task shown to a player is one sample "plate".
    """

    points: np.ndarray
    labels: np.ndarray | None = None
    plate_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = _as_points(self.points)
        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=np.int64)
            if self.labels.shape != (len(self.points),):
                raise ValueError("labels must be one per point")
        if self.plate_id is None:
            self.plate_id = self.content_hash()

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_true_clusters(self) -> int | None:
        if self.labels is None:
            return None
        return int(len(set(self.labels.tolist()) - {NOISE}))

    def content_hash(self) -> str:
        """Stable hash of the point coordinates, used as an exact-match key."""
        rounded = np.round(self.points, 6)
        return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]

    def normalized(self) -> "Plate":
        """Return a copy rescaled into the unit square, preserving aspect ratio."""
        pts = self.points
        lo = pts.min(axis=0)
        span = float(np.max(pts.max(axis=0) - lo))
        if span <= 0:
            span = 1.0
        return Plate(
            points=(pts - lo) / span,
            labels=None if self.labels is None else self.labels.copy(),
            plate_id=self.plate_id,
            meta={**self.meta, "normalized_from": {"offset": lo.tolist(), "span": span}},
        )


@dataclass
class Candidate:
    """One proposed clustering of a plate, produced by one algorithm setting."""

    labels: np.ndarray
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    predicted_score: float | None = None
    true_score: float | None = None

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int64)

    @property
    def n_clusters(self) -> int:
        return int(len(set(self.labels.tolist()) - {NOISE}))

    @property
    def noise_fraction(self) -> float:
        if len(self.labels) == 0:
            return 0.0
        return float(np.mean(self.labels == NOISE))

    def describe(self) -> str:
        params = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.algorithm}({params})"


@dataclass
class Solution:
    """The answer the pipeline submits for a plate."""

    labels: np.ndarray
    n_clusters: int
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)
    predicted_score: float | None = None
    source: str = "search"  # "search" | "memory"
    memory_similarity: float | None = None
    polygons: list[list[list[float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def to_json(self, indent: int = 2) -> str:
        payload = asdict(self)
        payload["labels"] = self.labels.tolist()
        return json.dumps(payload, indent=indent, default=float)
