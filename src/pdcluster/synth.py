"""Synthetic Project-Discovery-like plates.

Real flow-cytometry scatter plots are not tidy isotropic blobs: populations are
stretched along one axis, vary wildly in density, sit on a haze of debris, and
sometimes touch each other.  The generator below reproduces those shapes so the
ranker is trained on plates that look like the ones it has to solve.
"""

from __future__ import annotations

import numpy as np

from .types import NOISE, Plate

CLUSTER_SHAPES = ("gaussian", "elongated", "comet", "crescent", "dense_core")


def _random_covariance(rng: np.random.Generator, scale: float, anisotropy: float) -> np.ndarray:
    angle = rng.uniform(0, np.pi)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    sx = scale
    sy = scale / anisotropy
    return rot @ np.diag([sx**2, sy**2]) @ rot.T


def _draw_cluster(
    rng: np.random.Generator,
    center: np.ndarray,
    size: int,
    shape: str,
    scale: float,
) -> np.ndarray:
    if shape == "gaussian":
        cov = _random_covariance(rng, scale, rng.uniform(1.0, 1.4))
        return rng.multivariate_normal(center, cov, size=size)

    if shape == "elongated":
        cov = _random_covariance(rng, scale, rng.uniform(2.5, 6.0))
        return rng.multivariate_normal(center, cov, size=size)

    if shape == "comet":
        # A dense head with a tail trailing off in one direction.
        head_n = max(1, int(size * 0.65))
        tail_n = max(1, size - head_n)
        cov = _random_covariance(rng, scale * 0.6, 1.2)
        head = rng.multivariate_normal(center, cov, size=head_n)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction) + 1e-12
        t = rng.power(0.6, size=tail_n)[:, None]
        tail = center + t * direction * scale * rng.uniform(3.0, 6.0)
        tail = tail + rng.normal(scale=scale * 0.45, size=tail.shape)
        return np.vstack([head, tail])

    if shape == "crescent":
        theta = rng.uniform(0, np.pi, size=size)
        radius = scale * 2.5 * (1 + rng.normal(scale=0.08, size=size))
        arc = np.stack([np.cos(theta) * radius, np.sin(theta) * radius], axis=1)
        angle = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return center + arc @ rot.T

    if shape == "dense_core":
        core_n = max(1, int(size * 0.8))
        halo_n = max(1, size - core_n)
        core = rng.multivariate_normal(center, _random_covariance(rng, scale * 0.4, 1.1), core_n)
        halo = rng.multivariate_normal(center, _random_covariance(rng, scale * 1.6, 1.3), halo_n)
        return np.vstack([core, halo])

    raise ValueError(f"unknown shape {shape!r}")


def make_plate(
    rng: np.random.Generator | int | None = None,
    n_clusters: int | None = None,
    n_points: int | None = None,
    noise_fraction: float | None = None,
    separation: float | None = None,
    allow_empty: bool = True,
    seed: int | None = None,
) -> Plate:
    """Generate one labelled plate.

    Args:
        rng: seed or generator, for reproducibility.
        n_clusters: number of populations; random 0-5 when omitted.
        n_points: total point count including noise.
        noise_fraction: share of points that are unclustered debris.
        separation: how far apart cluster centres are placed (higher = easier).
        allow_empty: permit plates that genuinely contain no cluster.
        seed: convenience alias for `rng`, so callers can write `make_plate(seed=3)`.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    if n_clusters is None:
        low = 0 if allow_empty else 1
        n_clusters = int(rng.choice(np.arange(low, 6), p=_cluster_prior(low)))
    if n_points is None:
        n_points = int(rng.integers(300, 1400))
    if noise_fraction is None:
        noise_fraction = float(rng.beta(1.6, 6.0) * 0.55)
    if separation is None:
        separation = float(rng.uniform(0.9, 2.6))

    n_noise = int(round(n_points * noise_fraction))
    n_clustered = max(0, n_points - n_noise)

    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    if n_clusters > 0 and n_clustered > 0:
        weights = rng.dirichlet(np.full(n_clusters, rng.uniform(0.7, 3.0)))
        sizes = np.maximum(12, (weights * n_clustered).astype(int))
        centers = _place_centers(rng, n_clusters, separation)
        for idx in range(n_clusters):
            shape = str(rng.choice(CLUSTER_SHAPES))
            scale = float(rng.uniform(0.045, 0.13))
            pts = _draw_cluster(rng, centers[idx], int(sizes[idx]), shape, scale)
            chunks.append(pts)
            labels.append(np.full(len(pts), idx, dtype=np.int64))

    if n_noise > 0:
        noise = rng.uniform(-0.05, 1.05, size=(n_noise, 2))
        chunks.append(noise)
        labels.append(np.full(n_noise, NOISE, dtype=np.int64))

    if not chunks:  # entirely empty plate
        chunks.append(rng.uniform(0, 1, size=(30, 2)))
        labels.append(np.full(30, NOISE, dtype=np.int64))

    points = np.vstack(chunks)
    all_labels = np.concatenate(labels)

    order = rng.permutation(len(points))
    points, all_labels = points[order], all_labels[order]

    plate = Plate(
        points=points,
        labels=all_labels,
        meta={
            "synthetic": True,
            "n_clusters": int(n_clusters),
            "noise_fraction": round(float(noise_fraction), 4),
            "separation": round(separation, 3),
        },
    )
    return plate.normalized()


def _cluster_prior(low: int) -> np.ndarray:
    """Cluster-count prior: 1-3 clusters are by far the most common in game."""
    full = {0: 0.06, 1: 0.24, 2: 0.30, 3: 0.22, 4: 0.12, 5: 0.06}
    p = np.array([full[k] for k in range(low, 6)], dtype=float)
    return p / p.sum()


def _place_centers(rng: np.random.Generator, k: int, separation: float) -> np.ndarray:
    """Place centres with rejection sampling so they are not on top of each other."""
    min_dist = 0.16 * separation
    centers: list[np.ndarray] = []
    for _ in range(k):
        for _attempt in range(200):
            cand = rng.uniform(0.12, 0.88, size=2)
            if all(np.linalg.norm(cand - c) >= min_dist for c in centers):
                centers.append(cand)
                break
        else:
            centers.append(rng.uniform(0.12, 0.88, size=2))
    return np.array(centers)


def make_dataset(
    n_plates: int = 120,
    seed: int = 0,
    **plate_kwargs,
) -> list[Plate]:
    """Generate a reproducible list of labelled plates."""
    rng = np.random.default_rng(seed)
    return [make_plate(rng, **plate_kwargs) for _ in range(n_plates)]
