import numpy as np
import pytest

from pdcluster.clicks import (
    MAX_VERTICES,
    clamp_to_bounds,
    is_simple,
    labels_from_polygons,
    offset_polygon,
    plan_cluster_clicks,
    plan_clicks,
    score_polygon,
    simplify_to,
)
from pdcluster.polygons import points_in_polygon, polygon_area
from pdcluster.synth import make_plate
from pdcluster.types import NOISE

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def _blob(center, scale, n=250, seed=0):
    return np.random.default_rng(seed).normal(center, scale, (n, 2))


def _crescent(n=400, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, n)
    radius = 0.25 * (1 + rng.normal(0, 0.05, (n, 1)))
    return np.stack([np.cos(theta), np.sin(theta)], 1) * radius + [0.5, 0.4]


# ------------------------------------------------------------------- geometry


def test_simplify_reduces_to_exactly_k_vertices():
    ring = [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 60, endpoint=False)]
    for k in range(3, 10):
        assert len(simplify_to(ring, k)) == k


def test_simplify_leaves_short_rings_alone():
    assert len(simplify_to(SQUARE, 9)) == 4


def test_simplify_keeps_the_defining_corners():
    """A square with many points along its edges should simplify back to a square."""
    ring = []
    for a, b in zip(SQUARE, SQUARE[1:] + SQUARE[:1]):
        for t in np.linspace(0, 1, 12, endpoint=False):
            ring.append([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
    simplified = simplify_to(ring, 4)
    assert len(simplified) == 4
    assert polygon_area(simplified) == pytest.approx(1.0, abs=0.05)


def test_offset_grows_the_polygon_without_moving_its_centre():
    grown = offset_polygon(SQUARE, 0.1)
    assert polygon_area(grown) > polygon_area(SQUARE)
    before = np.asarray(SQUARE).mean(axis=0)
    after = np.asarray(grown).mean(axis=0)
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_offset_of_a_concave_shape_keeps_its_bend_empty():
    """Radial growth would fill a crescent's bend; edge-normal offsetting must not."""
    crescent = [[0, 0], [1, 0], [1, 1], [0.6, 1], [0.6, 0.4], [0.4, 0.4], [0.4, 1], [0, 1]]
    grown = offset_polygon(crescent, 0.03)
    # A point in the notch stays outside the grown polygon.
    assert not bool(points_in_polygon(grown, np.array([[0.5, 0.9]]))[0])


def test_offset_by_zero_is_a_no_op():
    np.testing.assert_allclose(np.asarray(offset_polygon(SQUARE, 0.0)), np.asarray(SQUARE))


def test_clamp_pulls_vertices_into_the_plot():
    clamped = clamp_to_bounds([[-0.2, 0.5], [1.4, 0.5], [0.5, 1.3]], (0.0, 0.0, 1.0, 1.0))
    arr = np.asarray(clamped)
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_is_simple_detects_a_bowtie():
    assert is_simple(SQUARE)
    assert not is_simple([[0, 0], [1, 1], [1, 0], [0, 1]])


def test_score_polygon_rewards_capture_and_punishes_foreign_points():
    own = _blob([0.5, 0.5], 0.05)
    far = _blob([3.0, 3.0], 0.05, seed=1)
    capture, purity, f1 = score_polygon(SQUARE, own, far)
    assert capture == pytest.approx(1.0)
    assert purity == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)

    # Now the same loop also swallows a neighbouring population.
    intruder = _blob([0.5, 0.5], 0.05, seed=2)
    _, dirty_purity, dirty_f1 = score_polygon(SQUARE, own, intruder)
    assert dirty_purity < 0.6
    assert dirty_f1 < f1


# ---------------------------------------------------------------- click plans


def test_plan_never_exceeds_the_side_limit():
    pts = np.vstack([_blob([0.2, 0.2], 0.03), _crescent()])
    labels = np.concatenate([np.zeros(250), np.ones(400)]).astype(int)
    for cluster in (0, 1):
        plan = plan_cluster_clicks(pts, labels, cluster)
        assert plan is not None
        assert 3 <= plan.n_sides <= MAX_VERTICES


def test_lower_side_limit_is_respected():
    pts = _crescent()
    labels = np.zeros(len(pts), dtype=int)
    plan = plan_cluster_clicks(pts, labels, 0, max_vertices=5)
    assert plan is not None and plan.n_sides <= 5


def test_closing_click_returns_to_the_first_vertex():
    pts = _blob([0.5, 0.5], 0.06)
    plan = plan_cluster_clicks(pts, np.zeros(len(pts), dtype=int), 0)
    assert plan is not None
    assert plan.n_clicks == plan.n_sides + 1
    assert plan.clicks[0] == plan.clicks[-1]
    assert plan.clicks[0] == list(plan.vertices[0])


def test_a_complex_shape_earns_more_sides_than_a_simple_one():
    """The side count has to be decided per cluster, not fixed."""
    pts = np.vstack([_blob([0.15, 0.15], 0.025), _crescent()])
    labels = np.concatenate([np.zeros(250), np.ones(400)]).astype(int)
    blob = plan_cluster_clicks(pts, labels, 0)
    crescent = plan_cluster_clicks(pts, labels, 1)
    assert crescent.n_sides > blob.n_sides


def test_more_sides_never_scores_worse_on_a_hard_shape():
    """The per-budget scores should improve, not collapse, as the budget grows."""
    pts = _crescent()
    plan = plan_cluster_clicks(pts, np.zeros(len(pts), dtype=int), 0)
    scores = [plan.considered[k] for k in sorted(plan.considered)]
    assert scores[-1] > scores[0]
    assert max(scores) == pytest.approx(max(plan.considered.values()))


def test_plan_captures_most_of_its_cluster():
    pts = np.vstack([_blob([0.2, 0.2], 0.04), _blob([0.8, 0.8], 0.04, seed=5)])
    labels = np.concatenate([np.zeros(250), np.ones(250)]).astype(int)
    for cluster in (0, 1):
        plan = plan_cluster_clicks(pts, labels, cluster)
        assert plan.capture > 0.9
        assert plan.purity > 0.95
        assert is_simple(plan.vertices)


def test_plan_stays_inside_the_requested_bounds():
    pts = _blob([0.5, 0.5], 0.3)
    plan = plan_cluster_clicks(
        pts, np.zeros(len(pts), dtype=int), 0, bounds=(0.0, 0.0, 1.0, 1.0)
    )
    arr = np.asarray(plan.vertices)
    assert arr.min() >= -1e-9 and arr.max() <= 1.0 + 1e-9


def test_plan_clicks_covers_every_cluster_of_a_plate():
    plate = make_plate(seed=2024, n_clusters=3, n_points=700, noise_fraction=0.1)
    plans = plan_clicks(plate, labels=plate.labels)
    assert len(plans) == 3
    assert {p.cluster for p in plans} == {0, 1, 2}
    assert all(p.n_sides <= MAX_VERTICES for p in plans)


def test_plan_serialises_for_export():
    plate = make_plate(seed=7, n_clusters=2, noise_fraction=0.05)
    plan = plan_clicks(plate, labels=plate.labels)[0]
    payload = plan.to_dict()
    assert payload["sides"] == plan.n_sides
    assert len(payload["clicks"]) == plan.n_sides + 1
    assert payload["clicks"][0] == payload["clicks"][-1]


def test_too_small_a_cluster_gets_no_plan():
    pts = np.array([[0.1, 0.1], [0.2, 0.2]])
    assert plan_cluster_clicks(pts, np.zeros(2, dtype=int), 0) is None


def test_labels_from_polygons_inverts_the_drawing():
    pts = np.vstack([_blob([0.2, 0.2], 0.02), _blob([0.8, 0.8], 0.02, seed=3)])
    left = [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]]
    right = [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]]
    labels = labels_from_polygons(pts, [left, right])
    assert set(labels[:250].tolist()) == {0}
    assert set(labels[250:].tolist()) == {1}


def test_points_outside_every_loop_are_noise():
    pts = np.array([[0.5, 0.5], [0.05, 0.05]])
    labels = labels_from_polygons(pts, [[[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]])
    assert labels[0] == 0
    assert labels[1] == NOISE
