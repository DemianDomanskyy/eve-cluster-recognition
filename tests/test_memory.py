import numpy as np
import pytest

from pdcluster.memory import ClusterMemory, fingerprint
from pdcluster.synth import make_plate
from pdcluster.types import Plate


@pytest.fixture()
def memory(tmp_path):
    mem = ClusterMemory(tmp_path / "mem.db")
    yield mem
    mem.close()


def test_exact_recall_returns_stored_labels(memory):
    plate = make_plate(seed=1, n_clusters=3)
    memory.remember(plate, plate.labels, score=1.0, algorithm="test", perfect=True)

    hit = memory.recall(plate)
    assert hit is not None
    assert hit.kind == "exact"
    assert hit.similarity == 1.0
    np.testing.assert_array_equal(hit.labels, plate.labels)


def test_unknown_plate_is_a_miss(memory):
    memory.remember(make_plate(seed=1, n_clusters=3), make_plate(seed=1, n_clusters=3).labels,
                    score=1.0, algorithm="test", perfect=True)
    assert memory.recall(make_plate(seed=999, n_clusters=2)) is None


def test_imperfect_solutions_are_not_stored_by_default(memory):
    plate = make_plate(seed=2, n_clusters=2)
    assert memory.remember(plate, plate.labels, score=0.8, algorithm="test", perfect=False) is None
    assert len(memory) == 0


def test_imperfect_can_be_stored_when_opted_in(tmp_path):
    mem = ClusterMemory(tmp_path / "m.db", store_only_perfect=False)
    plate = make_plate(seed=2, n_clusters=2)
    assert mem.remember(plate, plate.labels, score=0.8, algorithm="t", perfect=False) is not None
    assert len(mem) == 1
    mem.close()


def test_similar_plate_recalls_via_label_transfer(memory):
    """A jittered copy of a solved plate should still be recognised."""
    plate = make_plate(seed=5, n_clusters=3, noise_fraction=0.05)
    memory.remember(plate, plate.labels, score=1.0, algorithm="test", perfect=True)

    rng = np.random.default_rng(0)
    jittered = Plate(points=plate.points + rng.normal(scale=0.001, size=plate.points.shape))
    assert jittered.content_hash() != plate.content_hash()

    hit = memory.recall(jittered)
    assert hit is not None
    assert hit.kind == "similar"
    assert hit.similarity >= memory.min_similarity
    # The transfer must reproduce the same populations.
    agreement = float(np.mean(hit.labels == plate.labels))
    assert agreement > 0.95


def test_recall_counts_as_a_hit(memory):
    plate = make_plate(seed=3, n_clusters=1)
    memory.remember(plate, plate.labels, score=1.0, algorithm="test", perfect=True)
    memory.recall(plate)
    memory.recall(plate)
    assert memory.stats()["total_recalls"] == 2


def test_rewriting_keeps_the_better_score(tmp_path):
    mem = ClusterMemory(tmp_path / "m.db", store_only_perfect=False)
    plate = make_plate(seed=4, n_clusters=2)
    rid = mem.remember(plate, plate.labels, score=0.5, algorithm="weak", perfect=False)
    mem.remember(plate, plate.labels, score=0.9, algorithm="strong", perfect=False)
    assert mem.recall(plate).score == pytest.approx(0.9)
    mem.remember(plate, plate.labels, score=0.2, algorithm="worse", perfect=False)
    assert mem.recall(plate).score == pytest.approx(0.9)
    assert rid is not None and len(mem) == 1
    mem.close()


def test_forget_and_clear(memory):
    plate = make_plate(seed=6, n_clusters=2)
    rid = memory.remember(plate, plate.labels, score=1.0, algorithm="t", perfect=True)
    assert memory.forget(rid) is True
    assert memory.recall(plate) is None
    memory.remember(plate, plate.labels, score=1.0, algorithm="t", perfect=True)
    assert memory.clear() == 1


def test_capacity_evicts_least_used(tmp_path):
    mem = ClusterMemory(tmp_path / "m.db", max_entries=3)
    plates = [make_plate(seed=s, n_clusters=2) for s in range(5)]
    for p in plates:
        mem.remember(p, p.labels, score=1.0, algorithm="t", perfect=True)
    assert len(mem) <= 3
    mem.close()


def test_fingerprint_is_stable_and_normalised():
    plate = make_plate(seed=11, n_clusters=3)
    fp = fingerprint(plate)
    np.testing.assert_allclose(fp, fingerprint(plate))
    assert np.linalg.norm(fp) == pytest.approx(1.0, abs=1e-9)


def test_fingerprint_survives_translation_and_scale():
    plate = make_plate(seed=12, n_clusters=3)
    moved = Plate(points=plate.points * 7.5 + 100.0)
    assert float(fingerprint(plate) @ fingerprint(moved)) > 0.99


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "persist.db"
    plate = make_plate(seed=21, n_clusters=2)
    with ClusterMemory(path) as mem:
        mem.remember(plate, plate.labels, score=1.0, algorithm="t", perfect=True)
    with ClusterMemory(path) as reopened:
        assert len(reopened) == 1
        assert reopened.recall(plate) is not None
