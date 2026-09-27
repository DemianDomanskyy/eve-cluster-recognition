import numpy as np
import pytest

from pdcluster.candidates import generate_candidates
from pdcluster.features import FEATURE_NAMES, candidate_features, plate_features
from pdcluster.memory import ClusterMemory
from pdcluster.pipeline import ClusterRecognizer, evaluate
from pdcluster.ranker import CandidateRanker, heuristic_score
from pdcluster.scoring import score_clustering
from pdcluster.synth import make_dataset, make_plate
from pdcluster.types import NOISE, Plate


def _easy_plate(seed: int = 0, k: int = 3) -> Plate:
    """Three tight, far-apart blobs: any sane pipeline must get this exactly right."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.15, 0.15], [0.85, 0.2], [0.5, 0.85]])[:k]
    pts, labels = [], []
    for i, c in enumerate(centers):
        p = rng.normal(loc=c, scale=0.02, size=(120, 2))
        pts.append(p)
        labels.append(np.full(len(p), i))
    return Plate(points=np.vstack(pts), labels=np.concatenate(labels))


# ------------------------------------------------------------------ candidates


def test_candidates_are_generated_and_unique():
    plate = _easy_plate()
    cands = generate_candidates(plate)
    assert len(cands) > 10
    keys = {c.labels.tobytes() for c in cands}
    assert len(keys) == len(cands)


def test_candidate_pool_contains_the_right_answer():
    plate = _easy_plate()
    best = max(
        score_clustering(plate.labels, c.labels).score for c in generate_candidates(plate)
    )
    assert best == pytest.approx(1.0)


def test_empty_answer_is_always_available():
    plate = _easy_plate()
    assert any(c.algorithm == "empty" for c in generate_candidates(plate))


# -------------------------------------------------------------------- features


def test_feature_vector_is_complete_and_finite():
    plate = _easy_plate()
    pf = plate_features(plate)
    for cand in generate_candidates(plate)[:6]:
        feats = candidate_features(plate, cand, pf, stability_repeats=1)
        assert set(feats) == set(FEATURE_NAMES)
        assert all(np.isfinite(v) for v in feats.values())


def test_good_clustering_scores_above_bad_one_heuristically():
    plate = _easy_plate()
    pf = plate_features(plate)
    cands = {c.describe(): c for c in generate_candidates(plate)}
    good = next(c for d, c in cands.items() if d.startswith("kmeans(k=3)"))
    bad = next(c for d, c in cands.items() if d.startswith("kmeans(k=6)"))
    gf = candidate_features(plate, good, pf, stability_repeats=2)
    bf = candidate_features(plate, bad, pf, stability_repeats=2)
    assert heuristic_score(gf) > heuristic_score(bf)


# -------------------------------------------------------------------- pipeline


def test_solve_finds_three_obvious_clusters(tmp_path):
    plate = _easy_plate()
    rec = ClusterRecognizer(memory=ClusterMemory(tmp_path / "m.db"))
    result = rec.solve(plate)
    assert result.solution.n_clusters == 3
    assert result.report is not None and result.report.is_perfect


def test_perfect_solution_is_written_to_memory(tmp_path):
    plate = _easy_plate(seed=3)
    mem = ClusterMemory(tmp_path / "m.db")
    rec = ClusterRecognizer(memory=mem)
    result = rec.solve(plate)
    assert result.report.is_perfect
    assert len(mem) == 1


def test_second_solve_comes_from_memory(tmp_path):
    plate = _easy_plate(seed=4)
    rec = ClusterRecognizer(memory=ClusterMemory(tmp_path / "m.db"))
    first = rec.solve(plate)
    assert first.solution.source == "search"
    second = rec.solve(plate)
    assert second.solution.source == "memory"
    assert second.considered == 0
    np.testing.assert_array_equal(first.solution.labels, second.solution.labels)


def test_memory_can_be_bypassed(tmp_path):
    plate = _easy_plate(seed=5)
    rec = ClusterRecognizer(memory=ClusterMemory(tmp_path / "m.db"))
    rec.solve(plate)
    again = rec.solve(plate, allow_memory=False)
    assert again.solution.source == "search"


def test_teach_stores_a_human_verified_answer(tmp_path):
    plate = _easy_plate(seed=6)
    mem = ClusterMemory(tmp_path / "m.db")
    rec = ClusterRecognizer(memory=mem)
    rec.teach(plate, plate.labels)
    result = rec.solve(plate)
    assert result.solution.source == "memory"
    assert result.solution.algorithm == "taught"


def test_solution_carries_one_polygon_per_cluster():
    plate = _easy_plate()
    rec = ClusterRecognizer(use_memory=False)
    result = rec.solve(plate, polygons=True)
    assert len(result.solution.polygons) == result.solution.n_clusters
    assert all(len(p) >= 3 for p in result.solution.polygons)


def test_solution_serialises_to_json():
    plate = _easy_plate()
    result = ClusterRecognizer(use_memory=False).solve(plate)
    import json

    payload = json.loads(result.solution.to_json())
    assert len(payload["labels"]) == plate.n_points
    assert payload["n_clusters"] == 3


def test_evaluate_reports_reasonable_quality(tmp_path):
    plates = make_dataset(n_plates=6, seed=42, allow_empty=False)
    rec = ClusterRecognizer(memory=ClusterMemory(tmp_path / "m.db"), stability_repeats=1)
    report = evaluate(rec, plates)
    assert report.n_plates == 6
    assert 0.0 <= report.mean_score <= 1.0
    assert report.memory_hits == 0  # cold memory on the first pass


def test_memory_makes_the_second_pass_faster_and_no_worse(tmp_path):
    plates = [_easy_plate(seed=s) for s in range(4)]
    rec = ClusterRecognizer(memory=ClusterMemory(tmp_path / "m.db"), stability_repeats=1)
    first = evaluate(rec, plates)
    second = evaluate(rec, plates)
    assert second.memory_hits == len(plates)
    assert second.mean_score >= first.mean_score


# --------------------------------------------------------------------- ranker


def test_untrained_ranker_uses_the_heuristic():
    ranker = CandidateRanker()
    assert not ranker.is_trained
    assert 0.0 <= ranker.predict({n: 0.5 for n in FEATURE_NAMES}) <= 1.0


def test_ranker_round_trips_through_disk(tmp_path):
    from pdcluster.ranker import train_ranker

    plates = make_dataset(n_plates=5, seed=7, allow_empty=False)
    ranker, report = train_ranker(plates, stability_repeats=1)
    assert ranker.is_trained
    assert report.n_examples > 50

    path = ranker.save(tmp_path / "r.joblib")
    reloaded = CandidateRanker.load(path)
    feats = {n: 0.5 for n in FEATURE_NAMES}
    assert reloaded.predict(feats) == pytest.approx(ranker.predict(feats))


def test_synthetic_plates_are_labelled_and_normalised():
    for plate in make_dataset(n_plates=5, seed=1):
        assert plate.labels is not None
        assert plate.points.min() >= -1e-9
        assert plate.points.max() <= 1.0 + 1e-9
        assert set(plate.labels.tolist()) <= set(range(6)) | {NOISE}
