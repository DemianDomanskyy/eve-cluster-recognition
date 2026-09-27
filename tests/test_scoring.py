import numpy as np
import pytest

from pdcluster.scoring import iou_matrix, is_perfect, score_clustering
from pdcluster.types import NOISE


def test_identical_labelling_is_perfect():
    true = np.array([0, 0, 1, 1, 2, 2])
    assert is_perfect(true, true.copy())
    assert score_clustering(true, true.copy()).score == pytest.approx(1.0)


def test_label_permutation_is_still_perfect():
    """Cluster ids are arbitrary names, so a relabelling must not be punished."""
    true = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([2, 2, 0, 0, 1, 1])
    assert is_perfect(true, permuted)


def test_missing_cluster_is_penalised():
    true = np.array([0, 0, 0, 1, 1, 1])
    merged = np.array([0, 0, 0, 0, 0, 0])
    report = score_clustering(true, merged)
    assert not report.is_perfect
    assert not report.cluster_count_correct
    assert report.score < 0.6


def test_split_cluster_is_penalised():
    true = np.zeros(8, dtype=int)
    split = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    report = score_clustering(true, split)
    assert report.n_pred == 2 and report.n_true == 1
    assert not report.is_perfect


def test_empty_reference_rewards_empty_answer():
    true = np.full(10, NOISE)
    assert score_clustering(true, np.full(10, NOISE)).score == pytest.approx(1.0)
    assert score_clustering(true, np.zeros(10, dtype=int)).score < 1.0


def test_no_prediction_against_real_clusters_scores_zero():
    true = np.array([0, 0, 1, 1])
    report = score_clustering(true, np.full(4, NOISE))
    assert report.score == 0.0
    assert report.unmatched_true == 2


def test_near_miss_is_not_perfect():
    true = np.array([0] * 50 + [1] * 50)
    almost = true.copy()
    almost[:5] = 1  # 5 points on the wrong side
    report = score_clustering(true, almost)
    assert report.cluster_count_correct
    assert not report.is_perfect
    assert 0.8 < report.score < 1.0


def test_iou_matrix_shape_and_values():
    true = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    mat = iou_matrix(true, pred)
    assert mat.shape == (2, 2)
    assert mat[0, 0] == pytest.approx(1.0)
    assert mat[0, 1] == pytest.approx(0.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        score_clustering(np.array([0, 1]), np.array([0, 1, 1]))
