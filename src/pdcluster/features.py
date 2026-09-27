"""Features describing how good a clustering looks, without any ground truth.

The ranker never sees the reference answer at solve time, so every feature here
is computable from the point cloud plus the proposed labels alone.  They fall
into three groups: classical validity indices (silhouette and friends), geometric
sanity checks (cluster balance, density contrast, ellipse fit), and a stability
probe that re-runs the clustering on subsamples - the single most informative
signal, because a clustering that survives resampling is usually the real one.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .types import NOISE, Candidate, Plate

FEATURE_NAMES: list[str] = [
    "n_points_log",
    "n_clusters",
    "noise_fraction",
    "silhouette",
    "calinski_log",
    "davies_bouldin",
    "size_entropy",
    "size_ratio",
    "mean_intra_nn",
    "min_center_gap",
    "gap_ratio",
    "density_contrast",
    "ellipse_fit",
    "boundary_sharpness",
    "stability_ari",
    "hopkins",
    "knn_cv",
    "coverage",
]


def plate_features(plate: Plate, rng: np.random.Generator | None = None) -> dict[str, float]:
    """Features of the plate itself, shared by all of its candidates."""
    rng = rng or np.random.default_rng(0)
    points = plate.points
    n = len(points)
    k = min(5, max(1, n - 1))
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    nn = dists[:, 1] if dists.shape[1] > 1 else np.zeros(n)
    return {
        "n_points_log": float(np.log10(max(n, 1))),
        "hopkins": _hopkins(points, tree, rng),
        "knn_cv": float(np.std(nn) / (np.mean(nn) + 1e-12)),
    }


def _hopkins(points: np.ndarray, tree: cKDTree, rng: np.random.Generator) -> float:
    """Hopkins statistic: about 0.5 means uniform, closer to 1 means clustered."""
    n = len(points)
    m = max(5, min(80, n // 10))
    if n <= m:
        return 0.5
    lo, hi = points.min(axis=0), points.max(axis=0)
    uniform = rng.uniform(lo, hi, size=(m, 2))
    u_dist, _ = tree.query(uniform, k=1)
    idx = rng.choice(n, size=m, replace=False)
    w_dist, _ = tree.query(points[idx], k=2)
    w = w_dist[:, 1]
    total = float(np.sum(u_dist) + np.sum(w))
    return float(np.sum(u_dist) / total) if total > 0 else 0.5


def candidate_features(
    plate: Plate,
    candidate: Candidate,
    plate_feats: dict[str, float] | None = None,
    stability_repeats: int = 2,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute the full feature vector for one candidate."""
    rng = rng or np.random.default_rng(0)
    points = plate.points
    labels = candidate.labels
    feats = dict(plate_feats or plate_features(plate, rng))

    clustered = labels != NOISE
    uniq = sorted(set(labels[clustered].tolist()))
    k = len(uniq)

    feats["n_clusters"] = float(k)
    feats["noise_fraction"] = float(np.mean(~clustered))
    feats["coverage"] = float(np.mean(clustered))

    # An empty answer has no internal geometry to measure; neutral values keep it
    # comparable instead of letting it silently win on a vector of zeros.
    if k == 0:
        feats.update(
            silhouette=0.0,
            calinski_log=0.0,
            davies_bouldin=3.0,
            size_entropy=0.0,
            size_ratio=0.0,
            mean_intra_nn=0.0,
            min_center_gap=0.0,
            gap_ratio=0.0,
            density_contrast=1.0,
            ellipse_fit=0.0,
            boundary_sharpness=0.0,
            stability_ari=0.0,
        )
        return {name: float(feats.get(name, 0.0)) for name in FEATURE_NAMES}

    sizes = np.array([np.count_nonzero(labels == u) for u in uniq], dtype=float)
    probs = sizes / sizes.sum()
    feats["size_entropy"] = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(max(k, 2)))
    feats["size_ratio"] = float(sizes.min() / sizes.max())

    sub = points[clustered]
    sublab = labels[clustered]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if k >= 2 and len(sub) > k:
            idx = _subsample_idx(len(sub), 800, rng)
            try:
                feats["silhouette"] = float(silhouette_score(sub[idx], sublab[idx]))
            except ValueError:
                feats["silhouette"] = 0.0
            try:
                feats["calinski_log"] = float(np.log1p(calinski_harabasz_score(sub, sublab)))
                feats["davies_bouldin"] = float(davies_bouldin_score(sub, sublab))
            except ValueError:
                feats["calinski_log"], feats["davies_bouldin"] = 0.0, 3.0
        else:
            feats["silhouette"] = 0.0
            feats["calinski_log"] = 0.0
            feats["davies_bouldin"] = 1.0

    centers = np.array([points[labels == u].mean(axis=0) for u in uniq])
    intra: list[float] = []
    ellipse: list[float] = []
    for u in uniq:
        pts = points[labels == u]
        if len(pts) > 2:
            tree = cKDTree(pts)
            d, _ = tree.query(pts, k=2)
            intra.append(float(np.mean(d[:, 1])))
            ellipse.append(_ellipse_fit(pts))
    feats["mean_intra_nn"] = float(np.mean(intra)) if intra else 0.0
    feats["ellipse_fit"] = float(np.mean(ellipse)) if ellipse else 0.0

    if k >= 2:
        gaps = [
            float(np.linalg.norm(centers[i] - centers[j]))
            for i in range(k)
            for j in range(i + 1, k)
        ]
        feats["min_center_gap"] = float(min(gaps))
        feats["gap_ratio"] = float(min(gaps) / (feats["mean_intra_nn"] + 1e-12))
    else:
        feats["min_center_gap"] = 1.0
        feats["gap_ratio"] = 10.0

    feats["density_contrast"] = _density_contrast(points, labels)
    feats["boundary_sharpness"] = _boundary_sharpness(points, labels, centers, uniq)
    feats["stability_ari"] = _stability(plate, candidate, stability_repeats, rng)

    return {name: float(feats.get(name, 0.0)) for name in FEATURE_NAMES}


def _subsample_idx(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return rng.choice(n, size=cap, replace=False)


def _ellipse_fit(pts: np.ndarray) -> float:
    """How Gaussian a cluster is: the fraction inside its 95% error ellipse."""
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T) + np.eye(2) * 1e-9
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0
    d = pts - mu
    m2 = np.einsum("ij,jk,ik->i", d, inv, d)
    return float(np.mean(m2 <= 5.991))  # chi-square with 2 dof, 95% quantile


def _density_contrast(points: np.ndarray, labels: np.ndarray) -> float:
    """Ratio of local density inside clusters to local density in the noise region."""
    k = min(6, max(1, len(points) - 1))
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1)
    local = 1.0 / (np.mean(d[:, 1:], axis=1) + 1e-12)
    inside = labels != NOISE
    if not inside.any() or inside.all():
        return 1.0
    return float(np.mean(local[inside]) / (np.mean(local[~inside]) + 1e-12))


def _boundary_sharpness(
    points: np.ndarray, labels: np.ndarray, centers: np.ndarray, uniq: list[int]
) -> float:
    """Low values mean a cluster edge was drawn through a still-dense region."""
    scores: list[float] = []
    for i, u in enumerate(uniq):
        pts = points[labels == u]
        if len(pts) < 3:
            continue
        r_own = np.linalg.norm(pts - centers[i], axis=1)
        edge = float(np.quantile(r_own, 0.9))
        r_all = np.linalg.norm(points - centers[i], axis=1)
        inner = int(np.count_nonzero(r_own < edge))
        outer = int(np.count_nonzero((r_all > edge) & (r_all < edge * 1.5)))
        scores.append(float((inner + 1) / (inner + outer + 2)))
    return float(np.mean(scores)) if scores else 0.0


def _stability(
    plate: Plate, candidate: Candidate, repeats: int, rng: np.random.Generator
) -> float:
    """Re-cluster subsamples with the same algorithm and compare to this labelling.

    A clustering that reappears when 20% of the points are removed is far more
    likely to be the reference answer than one that dissolves.
    """
    if repeats <= 0:
        return 0.0
    # A single-cluster answer is trivially "stable": every point carries the same
    # label, so any re-clustering of a subsample agrees with it perfectly and the
    # ARI is 1.0 by construction.  That is not evidence of anything, and letting
    # it through hands k=1 a free top score on the feature the ranker leans on
    # most - which is exactly how a plate ends up with one loop around everything.
    if candidate.n_clusters < 2:
        return 0.0

    from .candidates import _relabel_compact  # local import avoids an import cycle

    fitter = _refit_fn(candidate)
    if fitter is None:
        return 0.0

    points = plate.points
    n = len(points)
    scores: list[float] = []
    for _ in range(repeats):
        idx = rng.choice(n, size=max(10, int(0.8 * n)), replace=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new = fitter(points[idx])
        except Exception:
            continue
        new = _relabel_compact(new, 1)
        scores.append(float(adjusted_rand_score(candidate.labels[idx], new)))
    return float(np.mean(scores)) if scores else 0.0


def _refit_fn(candidate: Candidate):
    """Rebuild the estimator behind a candidate so it can be applied to a subsample."""
    from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture

    p = candidate.params
    algo = candidate.algorithm
    if algo == "kmeans":
        return lambda x: KMeans(n_clusters=p["k"], n_init=4, random_state=1).fit_predict(x)
    if algo == "gmm":
        return lambda x: GaussianMixture(
            n_components=p["k"], covariance_type=p["covariance"], random_state=1
        ).fit_predict(x)
    if algo == "agglomerative":
        return lambda x: AgglomerativeClustering(
            n_clusters=p["k"], linkage=p["linkage"]
        ).fit_predict(x)
    if algo == "dbscan":
        return lambda x: DBSCAN(eps=p["eps"], min_samples=p["min_samples"]).fit_predict(x)
    if algo == "hdbscan":
        return lambda x: HDBSCAN(
            min_cluster_size=max(5, int(p["fraction"] * len(x)))
        ).fit_predict(x)
    return None


def feature_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)
