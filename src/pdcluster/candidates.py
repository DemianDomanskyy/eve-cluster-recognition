"""Candidate generation: propose many clusterings, let the ranker choose.

No single clustering algorithm wins on every plate.  K-means nails well-separated
round blobs but shatters elongated populations; DBSCAN and HDBSCAN handle odd
shapes and debris but are sensitive to their density parameters; a Gaussian
mixture is ideal for overlapping ellipses.  So we run a bounded sweep over all
of them and treat picking the winner as a learned ranking problem.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from .types import NOISE, Candidate, Plate

MAX_K = 6
"""Upper bound on cluster count; Project Discovery plates rarely exceed this."""


def _knn_distance_scale(points: np.ndarray, k: int = 4) -> np.ndarray:
    """Distances to the k-th nearest neighbour, the basis for DBSCAN's eps grid."""
    k = min(k, max(1, len(points) - 1))
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    return np.sort(dists[:, -1])


def _relabel_compact(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Renumber clusters 0..m-1 and demote undersized clusters to noise."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    out = np.full(labels.shape, NOISE, dtype=np.int64)
    next_id = 0
    for lab in sorted(set(labels.tolist()) - {NOISE}):
        mask = labels == lab
        if np.count_nonzero(mask) >= min_size:
            out[mask] = next_id
            next_id += 1
    return out


def generate_candidates(
    plate: Plate,
    max_k: int = MAX_K,
    min_cluster_size: int | None = None,
    include_empty: bool = True,
) -> list[Candidate]:
    """Produce the candidate pool for one plate, de-duplicated by labelling."""
    points = plate.points
    n = len(points)
    if min_cluster_size is None:
        min_cluster_size = max(8, int(0.015 * n))

    out: list[Candidate] = []

    def add(labels: np.ndarray, algorithm: str, **params) -> None:
        compact = _relabel_compact(labels, min_cluster_size)
        out.append(Candidate(labels=compact, algorithm=algorithm, params=params))

    if include_empty:
        # "There is nothing here" is a legitimate answer and must be rankable.
        add(np.full(n, NOISE), "empty")

    k_upper = min(max_k, max(1, n // max(min_cluster_size, 1)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for k in range(1, k_upper + 1):
            add(KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(points), "kmeans", k=k)

            for cov in ("full", "diag"):
                gmm = GaussianMixture(
                    n_components=k, covariance_type=cov, n_init=1, random_state=0
                ).fit(points)
                labels = gmm.predict(points)
                # Points in the far tails of every component are debris, not signal.
                logprob = gmm.score_samples(points)
                cut = np.quantile(logprob, 0.02)
                labels = np.where(logprob < cut, NOISE, labels)
                add(labels, "gmm", k=k, covariance=cov, bic=round(float(gmm.bic(points)), 2))

            if k >= 2:
                for linkage in ("ward", "average"):
                    add(
                        AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(points),
                        "agglomerative",
                        k=k,
                        linkage=linkage,
                    )

        kdist = _knn_distance_scale(points)
        for q in (0.55, 0.70, 0.82, 0.90, 0.95):
            eps = float(np.quantile(kdist, q))
            if eps <= 0:
                continue
            for min_samples in (5, 10, 20):
                if min_samples >= n:
                    continue
                add(
                    DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points),
                    "dbscan",
                    eps=round(eps, 5),
                    min_samples=min_samples,
                    eps_quantile=q,
                )

        for frac in (0.02, 0.04, 0.08, 0.15):
            mcs = max(5, int(frac * n))
            if mcs >= n // 2:
                continue
            add(
                HDBSCAN(min_cluster_size=mcs).fit_predict(points),
                "hdbscan",
                min_cluster_size=mcs,
                fraction=frac,
            )

    return _dedupe(out)


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Drop candidates whose labelling is identical to one already kept."""
    seen: dict[bytes, Candidate] = {}
    for cand in candidates:
        key = cand.labels.tobytes()
        if key not in seen:
            seen[key] = cand
    return list(seen.values())
