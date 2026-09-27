"""Turning a cluster outline into the clicks that draw it.

The in-game drawing mechanic is: every click drops a vertex, consecutive vertices
are joined by an edge, and clicking the first vertex again closes the loop.  A
polygon may have at most `MAX_VERTICES` sides, so a 300-vertex alpha shape is not
something you can actually draw - it has to be reduced to a handful of corners.

How many corners?  That is the interesting part, and it is not a fixed number: a
round blob is captured well by a hexagon, a comet with a long tail needs its
vertices spent differently, and a small tight cluster may be fine with a triangle.
So each vertex budget from 3 up to the maximum is tried and scored on what it
actually achieves:

* **capture** - the share of the cluster's own points that end up inside the loop
* **purity**  - the share of points inside the loop that belong to the cluster
                (enclosing a neighbouring population or a cloud of debris is what
                loses you points, so it has to cost something)

The two are combined as an F1 score, and the winner is the **fewest** vertices
that comes within `TOLERANCE` of the best score any budget achieved.  Spending
two more clicks for a 0.2% better outline is not worth it; spending them for a
15% better one is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .polygons import Polygon, cluster_polygons, points_in_polygon, polygon_area
from .types import NOISE, Plate, Solution

MAX_VERTICES = 9
"""Hard cap on polygon sides, matching the in-game limit."""

MIN_VERTICES = 3
TOLERANCE = 0.01
"""Give up this much score to save a vertex."""


# ------------------------------------------------------------------ geometry


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0)


def simplify_to(ring: Polygon, k: int) -> Polygon:
    """Reduce a ring to `k` vertices, dropping the least significant one each time.

    This is Visvalingam-Whyatt simplification: the vertex whose removal changes
    the enclosed area least is the one that matters least, so it goes first.  It
    keeps the corners that define the shape, which is what we want when the
    budget is nine clicks.
    """
    pts = [np.asarray(p, dtype=np.float64) for p in ring]
    if k < MIN_VERTICES or len(pts) <= k:
        return [[float(p[0]), float(p[1])] for p in pts]

    while len(pts) > k:
        n = len(pts)
        areas = [_triangle_area(pts[(i - 1) % n], pts[i], pts[(i + 1) % n]) for i in range(n)]
        pts.pop(int(np.argmin(areas)))
    return [[float(p[0]), float(p[1])] for p in pts]


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def is_simple(polygon: Polygon) -> bool:
    """True when no two non-adjacent edges of the polygon cross.

    Simplification can occasionally fold a polygon over itself, and a
    self-crossing loop is not a shape you can draw or submit.
    """
    n = len(polygon)
    if n < 4:
        return True
    for i in range(n):
        a1, a2 = polygon[i], polygon[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if _segments_cross(a1, a2, polygon[j], polygon[(j + 1) % n]):
                return False
    return True


def _signed_area(polygon: Polygon) -> float:
    arr = np.asarray(polygon, dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def offset_polygon(polygon: Polygon, distance: float) -> Polygon:
    """Push every edge outward along its own normal by `distance`.

    Not the same as scaling away from the centroid: a crescent's centroid lies
    outside the crescent, so radial growth shears the shape and swallows whatever
    is in the bend.  Offsetting each edge along its own normal and re-intersecting
    the neighbouring edges (a miter join) grows the loop while keeping its shape.
    """
    if distance <= 0 or len(polygon) < 3:
        return [list(v) for v in polygon]

    pts = np.asarray(polygon, dtype=np.float64)
    if _signed_area(polygon) < 0:  # work in counter-clockwise order
        pts = pts[::-1]
    n = len(pts)

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-12:
            lines.append((a, np.array([1.0, 0.0])))
            continue
        direction = edge / length
        outward = np.array([direction[1], -direction[0]])  # right of travel, i.e. outside
        lines.append((a + outward * distance, direction))

    out: list[list[float]] = []
    for i in range(n):
        p0, d0 = lines[(i - 1) % n]
        p1, d1 = lines[i]
        denominator = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(denominator) < 1e-9:  # parallel edges: no corner to miter
            out.append([float(p1[0]), float(p1[1])])
            continue
        t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / denominator
        corner = p0 + d0 * t
        # A spike means a near-reflex corner blew up; clamp it back to the edge.
        if float(np.linalg.norm(corner - pts[i])) > distance * 6.0:
            corner = pts[i] + (corner - pts[i]) / (
                np.linalg.norm(corner - pts[i]) + 1e-12
            ) * distance
        out.append([float(corner[0]), float(corner[1])])
    return out


# --------------------------------------------------------------- click plans


@dataclass
class ClickPlan:
    """A drawable polygon plus the exact click sequence that produces it."""

    cluster: int
    vertices: Polygon
    capture: float
    purity: float
    score: float
    considered: dict[int, float] = field(default_factory=dict)

    @property
    def n_sides(self) -> int:
        return len(self.vertices)

    @property
    def clicks(self) -> Polygon:
        """Vertices in click order, ending with the closing click on the first one."""
        if not self.vertices:
            return []
        return [list(v) for v in self.vertices] + [list(self.vertices[0])]

    @property
    def n_clicks(self) -> int:
        return len(self.clicks)

    def describe(self) -> str:
        return (
            f"cluster {self.cluster + 1}: {self.n_sides} sides, {self.n_clicks} clicks "
            f"(capture {self.capture:.1%}, purity {self.purity:.1%})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster + 1,
            "sides": self.n_sides,
            "clicks": self.clicks,
            "capture": round(self.capture, 4),
            "purity": round(self.purity, 4),
            "score": round(self.score, 4),
            "scores_by_vertex_count": {str(k): round(v, 4) for k, v in self.considered.items()},
        }


def clamp_to_bounds(polygon: Polygon, bounds: tuple[float, float, float, float]) -> Polygon:
    """Pull vertices back inside the plot area.

    Offsetting a loop outward can push a corner off the edge of the plot, and a
    click outside the plot area is not a click you can actually make.  Projecting
    the vertex back onto the boundary keeps the loop drawable.
    """
    x0, y0, x1, y1 = bounds
    return [
        [float(min(max(vx, x0), x1)), float(min(max(vy, y0), y1))] for vx, vy in polygon
    ]


def score_polygon(polygon: Polygon, own: np.ndarray, others: np.ndarray) -> tuple[float, float, float]:
    """Return (capture, purity, F1) for one candidate loop."""
    if len(own) == 0 or len(polygon) < 3:
        return 0.0, 0.0, 0.0
    inside_own = int(np.count_nonzero(points_in_polygon(polygon, own)))
    inside_other = (
        int(np.count_nonzero(points_in_polygon(polygon, others))) if len(others) else 0
    )
    capture = inside_own / len(own)
    purity = inside_own / max(inside_own + inside_other, 1)
    f1 = 0.0 if capture + purity == 0 else 2 * capture * purity / (capture + purity)
    return capture, purity, f1


def plan_cluster_clicks(
    points: np.ndarray,
    labels: np.ndarray,
    cluster: int,
    outline: Polygon | None = None,
    max_vertices: int = MAX_VERTICES,
    tolerance: float = TOLERANCE,
    bounds: tuple[float, float, float, float] | None = None,
) -> ClickPlan | None:
    """Work out how many sides this cluster needs, and where to click."""
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    own = points[labels == cluster]
    others = points[labels != cluster]
    if len(own) < 3:
        return None

    if outline is None:
        outlines = cluster_polygons(points, labels)
        ids = sorted(set(labels.tolist()) - {NOISE})
        if cluster not in ids:
            return None
        outline = outlines[ids.index(cluster)]
    if not outline or len(outline) < 3:
        return None

    if bounds is None:
        lo, hi = points.min(axis=0), points.max(axis=0)
        bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    span = float(np.max(own.max(axis=0) - own.min(axis=0))) or 1.0
    offsets = [0.0, span * 0.004, span * 0.01, span * 0.02, span * 0.04]

    def best_for(k: int) -> ClickPlan | None:
        """Best loop achievable with exactly k vertices, over the offset choices."""
        base = simplify_to(outline, k)
        if len(base) < 3:
            return None
        winner: ClickPlan | None = None
        for distance in offsets:
            candidate = offset_polygon(base, distance) if distance > 0 else base
            clamped = clamp_to_bounds(candidate, bounds)
            if is_simple(clamped):
                candidate = clamped
            if len(candidate) < 3 or not is_simple(candidate):
                continue
            capture, purity, f1 = score_polygon(candidate, own, others)
            if winner is None or f1 > winner.score:
                winner = ClickPlan(
                    cluster=cluster,
                    vertices=candidate,
                    capture=capture,
                    purity=purity,
                    score=f1,
                )
        return winner

    upper = max(MIN_VERTICES, min(max_vertices, MAX_VERTICES))
    by_k: dict[int, ClickPlan] = {}
    considered: dict[int, float] = {}
    for k in range(MIN_VERTICES, upper + 1):
        plan = best_for(k)
        if plan is not None:
            by_k[k] = plan
            considered[k] = plan.score

    if not by_k:
        return None

    top = max(considered.values())
    # Spend the fewest clicks that gets essentially the best result.
    chosen = min(k for k, score in considered.items() if score >= top - tolerance)
    best = by_k[chosen]
    best.considered = considered
    return best


def plan_clicks(
    plate: Plate,
    solution: Solution | None = None,
    labels: np.ndarray | None = None,
    max_vertices: int = MAX_VERTICES,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[ClickPlan]:
    """Plan the click sequence for every cluster in a solution."""
    if labels is None:
        if solution is None:
            raise ValueError("pass either a solution or labels")
        labels = solution.labels
    labels = np.asarray(labels)

    outlines: list[Polygon] | None = None
    if solution is not None and solution.polygons:
        outlines = solution.polygons

    ids = sorted(set(labels.tolist()) - {NOISE})
    plans: list[ClickPlan] = []
    for i, cluster in enumerate(ids):
        outline = outlines[i] if outlines and i < len(outlines) else None
        plan = plan_cluster_clicks(
            plate.points,
            labels,
            cluster,
            outline=outline,
            max_vertices=max_vertices,
            bounds=bounds,
        )
        if plan is not None:
            plans.append(plan)
    return plans


def labels_from_polygons(points: np.ndarray, polygons: list[Polygon]) -> np.ndarray:
    """Assign points to the polygon that encloses them - the inverse of drawing.

    Used to grade a loop somebody drew by hand: overlapping loops give the point
    to the smaller one, since that is the more specific claim.
    """
    points = np.asarray(points, dtype=np.float64)
    labels = np.full(len(points), NOISE, dtype=np.int64)
    order = sorted(range(len(polygons)), key=lambda i: -polygon_area(polygons[i]))
    for i in order:
        if len(polygons[i]) >= 3:
            labels[points_in_polygon(polygons[i], points)] = i
    return labels
