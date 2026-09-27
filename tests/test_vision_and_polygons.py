import numpy as np
import pytest

from pdcluster.loading import load_plate, save_dataset, save_plate
from pdcluster.plotting import cluster_table, format_cluster_table, plot_solution
from pdcluster.polygons import (
    cluster_polygons,
    convex_polygon,
    points_in_polygon,
    polygon_area,
    trim_outliers,
)
from pdcluster.synth import make_plate
from pdcluster.types import NOISE, Plate, Solution
from pdcluster.vision import points_from_image, render_plate_image


# -------------------------------------------------------------------- polygons


def _square(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, size=(n, 2))


def test_convex_polygon_wraps_its_points():
    pts = _square()
    poly = convex_polygon(pts)
    assert poly is not None and len(poly) >= 3
    assert polygon_area(poly) > 0.2


def test_polygon_contains_its_own_cluster():
    plate = make_plate(seed=8, n_clusters=2, noise_fraction=0.0)
    polys = cluster_polygons(plate.points, plate.labels, keep_quantile=1.0, pad=0.01)
    assert len(polys) == 2
    for lab, poly in zip(sorted(set(plate.labels.tolist()) - {NOISE}), polys):
        inside = points_in_polygon(poly, plate.points[plate.labels == lab])
        assert inside.mean() > 0.9


def test_trim_outliers_removes_a_stray_point():
    pts = np.vstack([_square(100), np.array([[50.0, 50.0]])])
    trimmed = trim_outliers(pts, keep_quantile=0.95)
    assert len(trimmed) < len(pts)
    assert trimmed.max() < 10.0


def test_concave_hull_is_tighter_than_convex_on_a_crescent():
    """An alpha shape should not claim the empty middle of a crescent."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, size=600)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * (
        1 + rng.normal(scale=0.02, size=(600, 1))
    )
    labels = np.zeros(len(pts), dtype=int)
    concave = cluster_polygons(pts, labels, keep_quantile=1.0, pad=0.0, concave=True)[0]
    convex = cluster_polygons(pts, labels, keep_quantile=1.0, pad=0.0, concave=False)[0]
    assert polygon_area(concave) < polygon_area(convex)


def test_no_polygon_for_noise_only_labels():
    pts = _square(50)
    assert cluster_polygons(pts, np.full(50, NOISE)) == []


# ---------------------------------------------------------------------- vision


def test_points_recovered_from_a_rendered_image(tmp_path):
    """Render a plate, read it back from pixels, and check the coordinates survive.

    Compared point-to-point rather than via cluster centroids: these plates contain
    elongated overlapping populations, so a centroid comparison would be measuring
    k-means, not the extractor.
    """
    from scipy.spatial import cKDTree

    plate = make_plate(seed=13, n_clusters=3, n_points=300, noise_fraction=0.0)
    img = render_plate_image(plate, tmp_path / "plate.png", size=1400, radius=1)

    recovered = points_from_image(img)
    assert recovered.n_points == plate.n_points

    # Every original marker should have a recovered point almost exactly on it.
    dist, _ = cKDTree(recovered.points).query(plate.points, k=1)
    assert float(np.max(dist)) < 0.01
    assert float(np.mean(dist)) < 0.002


def test_recovered_image_clusters_the_same_way(tmp_path):
    """A plate solved from pixels should get the same answer as from the raw data."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from scipy.spatial import cKDTree

    plate = make_plate(seed=13, n_clusters=3, n_points=300, noise_fraction=0.0)
    img = render_plate_image(plate, tmp_path / "plate.png", size=1400, radius=1)
    recovered = points_from_image(img)

    from_pixels = KMeans(n_clusters=3, n_init=5, random_state=0).fit_predict(recovered.points)
    from_data = KMeans(n_clusters=3, n_init=5, random_state=0).fit_predict(plate.points)
    # Align the two point orderings before comparing labels.
    _, order = cKDTree(recovered.points).query(plate.points, k=1)
    assert adjusted_rand_score(from_data, from_pixels[order]) > 0.95


def test_image_with_no_markers_returns_no_points(tmp_path):
    from PIL import Image

    path = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(path)
    assert points_from_image(path).n_points == 0


def test_loader_dispatches_on_extension(tmp_path):
    plate = make_plate(seed=14, n_clusters=2)

    npz = save_plate(plate, tmp_path / "p.npz")
    loaded = load_plate(npz)
    np.testing.assert_allclose(loaded.points, plate.points)
    np.testing.assert_array_equal(loaded.labels, plate.labels)

    csv = tmp_path / "p.csv"
    csv.write_text(
        "x,y,label\n" + "\n".join(f"{x},{y},{l}" for (x, y), l in zip(plate.points, plate.labels)),
        encoding="utf-8",
    )
    from_csv = load_plate(csv)
    np.testing.assert_allclose(from_csv.points, plate.points, atol=1e-9)
    np.testing.assert_array_equal(from_csv.labels, plate.labels)


def test_unsupported_format_raises(tmp_path):
    bad = tmp_path / "p.xyz"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_plate(bad)


def test_dataset_round_trip(tmp_path):
    from pdcluster.loading import load_dataset
    from pdcluster.synth import make_dataset

    plates = make_dataset(n_plates=4, seed=2)
    save_dataset(plates, tmp_path / "ds")
    reloaded = load_dataset(tmp_path / "ds")
    assert len(reloaded) == 4
    np.testing.assert_allclose(reloaded[0].points, plates[0].points)


# -------------------------------------------------------------------- plotting


def test_plot_writes_a_png(tmp_path):
    plate = make_plate(seed=15, n_clusters=3)
    solution = Solution(
        labels=plate.labels,
        n_clusters=3,
        algorithm="reference",
        polygons=cluster_polygons(plate.points, plate.labels),
    )
    out = plot_solution(plate, solution, tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 1000


def test_plot_handles_more_clusters_than_palette_slots(tmp_path):
    plate = make_plate(seed=16, n_clusters=5, noise_fraction=0.1)
    solution = Solution(labels=plate.labels, n_clusters=5, algorithm="reference")
    assert plot_solution(plate, solution, tmp_path / "many.png").exists()


def test_cluster_table_rows_and_text():
    plate = make_plate(seed=17, n_clusters=2, noise_fraction=0.1)
    solution = Solution(labels=plate.labels, n_clusters=2, algorithm="reference")
    rows = cluster_table(plate, solution)
    assert len(rows) == 3  # two clusters plus the unclustered row
    assert sum(int(r["points"]) for r in rows) == plate.n_points
    assert "cluster" in format_cluster_table(rows)
