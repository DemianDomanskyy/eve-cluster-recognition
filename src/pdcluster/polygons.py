"""Turning cluster labels into the polygon outlines a submission actually needs.

In game you do not hand in labels, you draw a loop around each population.  A
convex hull is the safe answer but it badly over-claims empty space on crescent
and comet shaped clusters - on a half-ring it can be eight times too large, and
every extra scrap of empty space is room for foreign points to fall inside your
loop.  So the outline is an alpha shape (a concave hull built from the Delaunay
triangulation), chosen by an explicit trade-off:

    take the smallest-area outline that still contains the cluster.

Alpha shapes form a family: tight alpha hugs the points but sheds the sparse
fringe, and as alpha grows the shape relaxes until it is exactly the convex hull.
So we walk that family from tight to loose, keep every outline that holds at
least `min_containment` of the points, and submit the smallest one.  The convex
hull is always in the running as the last resort, since it contains everything
by construction.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from .types import NOISE

Polygon = list[list[float]]

ALPHA_LADDER: tuple[float, ...] = (2.0, 2.8, 4.0, 6.0, 9.0, 14.0, 20.0, 30.0, 45.0, 70.0, 100.0)
"""Alpha factors (in units of mean nearest-neighbour spacing), tight to loose."""


def trim_outliers(pts: np.ndarray, keep_quantile: float = 0.98) -> np.ndarray:
    """Drop the furthest few points so one stray does not inflate the outline."""
    if len(pts) < 8 or keep_quantile >= 1.0:
        return pts
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T) + np.eye(2) * 1e-9
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return pts
    d = pts - mu
    m2 = np.einsum("ij,jk,ik->i", d, inv, d)
    cutoff = float(np.quantile(m2, keep_quantile))
    kept = pts[m2 <= cutoff]
    return kept if len(kept) >= 3 else pts


# ------------------------------------------------------------------ alpha shape


def _circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab, bc, ca = np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)
    s = (ab + bc + ca) / 2.0
    area_sq = s * (s - ab) * (s - bc) * (s - ca)
    if area_sq <= 1e-18:
        return float("inf")
    return float(ab * bc * ca / (4.0 * np.sqrt(area_sq)))


class _AlphaContext:
    """Delaunay triangulation plus per-triangle circumradii, computed once.

    Walking the alpha ladder means filtering the same triangulation at a dozen
    thresholds, so the expensive parts are shared across every rung.
    """

    def __init__(self, pts: np.ndarray) -> None:
        self.points = pts
        self.ok = False
        if len(pts) < 5:
            return
        try:
            tri = Delaunay(pts)
        except Exception:
            return
        self.simplices = tri.simplices
        self.radii = np.array(
            [_circumradius(pts[s[0]], pts[s[1]], pts[s[2]]) for s in self.simplices]
        )
        d, _ = cKDTree(pts).query(pts, k=2)
        self.spacing = float(np.mean(d[:, 1]))
        self.ok = self.spacing > 0 and len(self.simplices) > 0

    def ring_at(self, alpha_factor: float) -> Polygon | None:
        if not self.ok:
            return None
        alpha = self.spacing * alpha_factor
        keep = self.simplices[self.radii <= alpha]
        if len(keep) == 0:
            return None

        counts: dict[tuple[int, int], int] = {}
        for s in keep:
            for i, j in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
                key = (min(int(i), int(j)), max(int(i), int(j)))
                counts[key] = counts.get(key, 0) + 1
        # An edge shared by two kept triangles is interior; a lone edge is boundary.
        boundary = {e for e, c in counts.items() if c == 1}
        if len(boundary) < 3:
            return None
        return _trace_outer_ring(self.points, boundary)


def _trace_outer_ring(points: np.ndarray, edges: set[tuple[int, int]]) -> Polygon | None:
    """Walk the outer face of the boundary graph.

    Chaining edges greedily breaks apart wherever the boundary touches itself (a
    vertex with four boundary edges), which silently yields a fragment of the real
    outline.  Starting from the lowest vertex - guaranteed to be on the outer face -
    and always taking the tightest turn traces the true outer boundary instead.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    if not adjacency:
        return None

    start = min(adjacency, key=lambda i: (points[i][1], points[i][0]))
    ring = [start]
    prev: int | None = None
    current = start
    incoming = np.array([-1.0, 0.0])  # arrive at the lowest vertex travelling -x

    for _ in range(4 * len(edges) + 10):
        neighbours = [w for w in adjacency[current] if w != prev] or list(adjacency[current])
        if not neighbours:
            return None
        back = np.arctan2(-incoming[1], -incoming[0])
        best: int | None = None
        best_angle: float | None = None
        for w in neighbours:
            vec = points[w] - points[current]
            angle = float((np.arctan2(vec[1], vec[0]) - back) % (2 * np.pi))
            if best_angle is None or angle < best_angle:
                best, best_angle = w, angle
        if best is None:
            return None
        prev, incoming = current, points[best] - points[current]
        current = best
        if current == start:
            return [[float(points[i][0]), float(points[i][1])] for i in ring]
        ring.append(current)
    return None  # pragma: no cover - the walk is bounded by the edge count


def alpha_shape(pts: np.ndarray, alpha_factor: float = 4.0) -> Polygon | None:
    """Concave hull of a point set at one alpha, or None if none can be formed."""
    return _AlphaContext(np.asarray(pts, dtype=np.float64)).ring_at(alpha_factor)


def convex_polygon(pts: np.ndarray) -> Polygon | None:
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    return [[float(pts[i][0]), float(pts[i][1])] for i in hull.vertices]


def pad_polygon(polygon: Polygon, pad: float) -> Polygon:
    """Push vertices outward from the centroid, leaving a small safety margin."""
    if pad <= 0 or not polygon:
        return polygon
    arr = np.asarray(polygon, dtype=np.float64)
    centroid = arr.mean(axis=0)
    direction = arr - centroid
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr + direction / norms * pad).tolist()


def tightest_containing_polygon(
    pts: np.ndarray, min_containment: float = 0.97
) -> Polygon | None:
    """Smallest-area outline that still encloses `min_containment` of the points."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return None

    convex = convex_polygon(pts)
    context = _AlphaContext(pts)
    if not context.ok:
        return convex

    # Cluster points sit exactly *on* a candidate outline, where a ray-casting test
    # is a coin flip; testing against a hair-width expansion makes "on the
    # boundary" count as inside, which is what containment is meant to mean.
    epsilon = context.spacing * 0.25

    best: Polygon | None = convex
    best_area = polygon_area(convex) if convex else float("inf")

    for factor in ALPHA_LADDER:
        poly = context.ring_at(factor)
        if poly is None or len(poly) < 3:
            continue
        area = polygon_area(poly)
        if area >= best_area:
            continue
        containment = float(np.mean(points_in_polygon(pad_polygon(poly, epsilon), pts)))
        if containment >= min_containment:
            best, best_area = poly, area

    return best


def cluster_polygons(
    points: np.ndarray,
    labels: np.ndarray,
    keep_quantile: float = 0.98,
    pad: float = 0.005,
    concave: bool = True,
    min_containment: float = 0.97,
) -> list[Polygon]:
    """One outline per cluster, in ascending label order."""
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    out: list[Polygon] = []
    for lab in sorted(set(labels.tolist()) - {NOISE}):
        pts = trim_outliers(points[labels == lab], keep_quantile)
        poly = tightest_containing_polygon(pts, min_containment) if concave else convex_polygon(pts)
        if poly is None or len(poly) < 3:
            poly = convex_polygon(pts)
        if poly is None:
            continue
        out.append(pad_polygon(poly, pad))
    return out


def polygon_area(polygon: Polygon | None) -> float:
    """Shoelace area, useful for sanity-checking an outline."""
    if not polygon:
        return 0.0
    arr = np.asarray(polygon, dtype=np.float64)
    if len(arr) < 3:
        return 0.0
    x, y = arr[:, 0], arr[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def points_in_polygon(polygon: Polygon, points: np.ndarray) -> np.ndarray:
    """Even-odd ray-casting test, so a submitted outline can be verified."""
    arr = np.asarray(polygon, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    inside = np.zeros(len(pts), dtype=bool)
    n = len(arr)
    for i in range(n):
        x1, y1 = arr[i]
        x2, y2 = arr[(i + 1) % n]
        crosses = ((y1 > pts[:, 1]) != (y2 > pts[:, 1])) & (
            pts[:, 0] < (x2 - x1) * (pts[:, 1] - y1) / (y2 - y1 + 1e-18) + x1
        )
        inside ^= crosses
    return inside
