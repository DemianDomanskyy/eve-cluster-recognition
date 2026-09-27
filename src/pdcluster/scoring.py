"""Grading a proposed clustering against ground truth.

Project Discovery grades a submission by comparing the regions a player marked
against the consensus/reference answer: you need the right *number* of clusters
and each one has to overlap the reference closely.  We reproduce that with an
optimal one-to-one assignment (Hungarian algorithm) over pairwise Jaccard
overlap, plus an explicit penalty for inventing or missing clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import NOISE

PERFECT_IOU = 0.97
"""Per-cluster overlap above which a matched cluster counts as exactly right."""


@dataclass
class ScoreReport:
    score: float
    cluster_count_correct: bool
    n_true: int
    n_pred: int
    matched_iou: list[float] = field(default_factory=list)
    unmatched_true: int = 0
    unmatched_pred: int = 0
    noise_accuracy: float = 1.0

    @property
    def is_perfect(self) -> bool:
        """True when every reference cluster is recovered essentially exactly."""
        return (
            self.cluster_count_correct
            and self.unmatched_true == 0
            and self.unmatched_pred == 0
            and bool(self.matched_iou)
            and min(self.matched_iou) >= PERFECT_IOU
        )

    def summary(self) -> str:
        ious = ", ".join(f"{v:.3f}" for v in self.matched_iou) or "-"
        return (
            f"score={self.score:.4f} clusters={self.n_pred}/{self.n_true} "
            f"iou=[{ious}] perfect={self.is_perfect}"
        )


def _cluster_index(labels: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for lab in sorted(set(labels.tolist()) - {NOISE}):
        out[int(lab)] = labels == lab
    return out


def iou_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard overlap between reference and predicted clusters."""
    true_masks = _cluster_index(np.asarray(true_labels))
    pred_masks = _cluster_index(np.asarray(pred_labels))
    mat = np.zeros((len(true_masks), len(pred_masks)), dtype=np.float64)
    for i, tm in enumerate(true_masks.values()):
        for j, pm in enumerate(pred_masks.values()):
            union = np.count_nonzero(tm | pm)
            if union:
                mat[i, j] = np.count_nonzero(tm & pm) / union
    return mat


def score_clustering(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    count_penalty: float = 0.5,
) -> ScoreReport:
    """Score `pred_labels` against `true_labels` in [0, 1].

    The score is the mean overlap of optimally matched clusters, scaled down by
    `count_penalty` for each cluster that was hallucinated or missed.  Getting
    the cluster count wrong is the single most damaging mistake, which mirrors
    how the in-game grader behaves.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    if true_labels.shape != pred_labels.shape:
        raise ValueError("label arrays must be the same length")

    n_true = int(len(set(true_labels.tolist()) - {NOISE}))
    n_pred = int(len(set(pred_labels.tolist()) - {NOISE}))

    # Degenerate case: the reference says "no clusters at all".
    if n_true == 0:
        return ScoreReport(
            score=1.0 if n_pred == 0 else max(0.0, 1.0 - count_penalty * n_pred),
            cluster_count_correct=n_pred == 0,
            n_true=0,
            n_pred=n_pred,
            unmatched_pred=n_pred,
            noise_accuracy=float(np.mean(pred_labels == NOISE)),
        )
    if n_pred == 0:
        return ScoreReport(
            score=0.0,
            cluster_count_correct=False,
            n_true=n_true,
            n_pred=0,
            unmatched_true=n_true,
            noise_accuracy=float(np.mean(true_labels == NOISE)),
        )

    mat = iou_matrix(true_labels, pred_labels)
    rows, cols = linear_sum_assignment(-mat)
    matched = [float(mat[r, c]) for r, c in zip(rows, cols)]

    unmatched_true = n_true - len(matched)
    unmatched_pred = n_pred - len(matched)
    base = float(np.mean(matched)) if matched else 0.0
    penalty = count_penalty * (unmatched_true + unmatched_pred) / max(n_true, 1)
    score = float(np.clip(base - penalty, 0.0, 1.0))

    true_noise = true_labels == NOISE
    pred_noise = pred_labels == NOISE
    denom = int(np.count_nonzero(true_noise | pred_noise))
    noise_acc = 1.0 if denom == 0 else float(np.count_nonzero(true_noise & pred_noise) / denom)

    return ScoreReport(
        score=score,
        cluster_count_correct=n_true == n_pred,
        n_true=n_true,
        n_pred=n_pred,
        matched_iou=sorted(matched),
        unmatched_true=unmatched_true,
        unmatched_pred=unmatched_pred,
        noise_accuracy=noise_acc,
    )


def is_perfect(true_labels: np.ndarray, pred_labels: np.ndarray) -> bool:
    return score_clustering(true_labels, pred_labels).is_perfect
