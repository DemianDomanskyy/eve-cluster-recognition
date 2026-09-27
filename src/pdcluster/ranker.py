"""The learned component: a model that predicts a candidate's grade.

Training data is cheap because the synthetic generator knows the right answer.
For every plate we enumerate candidates, grade each one with `scoring`, and fit a
gradient-boosted regressor from the ground-truth-free feature vector to that
grade.  At solve time the highest predicted grade wins, so the system learns
*which algorithm to trust on which kind of plate* rather than being hard-wired
to one clustering method.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .candidates import generate_candidates
from .features import FEATURE_NAMES, extract_many, feature_vector, plate_features
from .scoring import score_clustering
from .types import Candidate, Plate

MODEL_VERSION = 1


def heuristic_score(feats: dict[str, float]) -> float:
    """Reasonable ranking with no trained model, used as a cold-start fallback.

    Leans on stability and silhouette, and distrusts answers that throw away most
    of the plate as noise or split it into many tiny unbalanced pieces.
    """
    s = 0.0
    s += 0.45 * feats.get("stability_ari", 0.0)
    s += 0.25 * max(0.0, feats.get("silhouette", 0.0))
    s += 0.10 * min(1.0, feats.get("gap_ratio", 0.0) / 8.0)
    s += 0.10 * feats.get("ellipse_fit", 0.0)
    s += 0.10 * feats.get("size_ratio", 0.0)
    s -= 0.25 * max(0.0, feats.get("noise_fraction", 0.0) - 0.45)
    s -= 0.04 * max(0.0, feats.get("n_clusters", 0.0) - 4.0)
    if feats.get("n_clusters", 0.0) == 0:
        s = 0.12 * (1.0 - abs(feats.get("hopkins", 0.5) - 0.5) * 2.0)
    return float(np.clip(s, 0.0, 1.0))


@dataclass
class TrainingReport:
    n_plates: int
    n_examples: int
    train_mae: float
    test_mae: float
    baseline_mae: float
    feature_importance: dict[str, float]

    def summary(self) -> str:
        top = sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:5]
        top_str = ", ".join(f"{k}={v:.3f}" for k, v in top)
        return (
            f"plates={self.n_plates} examples={self.n_examples} "
            f"test_mae={self.test_mae:.4f} (constant baseline {self.baseline_mae:.4f})\n"
            f"most useful features: {top_str}"
        )


class CandidateRanker:
    """Wraps the regressor plus the feature contract it was trained on."""

    def __init__(self, model=None, feature_names: list[str] | None = None) -> None:
        self.model = model
        self.feature_names = feature_names or list(FEATURE_NAMES)

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, feats: dict[str, float]) -> float:
        if self.model is None:
            return heuristic_score(feats)
        x = feature_vector(feats).reshape(1, -1)
        return float(np.clip(self.model.predict(x)[0], 0.0, 1.0))

    def rank(self, candidates: list[Candidate]) -> list[Candidate]:
        """Attach predicted scores and return candidates best-first."""
        for cand in candidates:
            cand.predicted_score = self.predict(cand.features)
        return sorted(candidates, key=lambda c: -(c.predicted_score or 0.0))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": MODEL_VERSION,
                "model": self.model,
                "feature_names": self.feature_names,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CandidateRanker":
        blob = joblib.load(Path(path))
        if blob.get("version") != MODEL_VERSION:
            raise ValueError(f"unsupported model version {blob.get('version')!r}")
        return cls(model=blob["model"], feature_names=blob["feature_names"])


def build_training_table(
    plates: list[Plate],
    stability_repeats: int = 2,
    seed: int = 0,
    progress: bool = False,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Enumerate and grade every candidate for every plate.

    Returns the feature matrix, the target grades, and per-row provenance so the
    table can be inspected or exported.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    provenance: list[dict] = []

    for i, plate in enumerate(plates):
        if plate.labels is None:
            raise ValueError("training plates must carry ground-truth labels")
        pf = plate_features(plate, rng)
        candidates = generate_candidates(plate)
        features = extract_many(
            plate, candidates, pf, stability_repeats=stability_repeats, seed=seed, n_jobs=n_jobs
        )
        for cand, feats in zip(candidates, features):
            cand.features = feats
            report = score_clustering(plate.labels, cand.labels)
            cand.true_score = report.score
            rows.append(feature_vector(cand.features))
            targets.append(report.score)
            provenance.append(
                {
                    "plate_id": plate.plate_id,
                    "algorithm": cand.algorithm,
                    "params": cand.params,
                    "score": report.score,
                    "perfect": report.is_perfect,
                }
            )
        if progress:
            print(f"  plate {i + 1}/{len(plates)}: {len(rows)} examples", flush=True)

    return np.array(rows), np.array(targets), provenance


def train_ranker(
    plates: list[Plate],
    stability_repeats: int = 2,
    seed: int = 0,
    test_size: float = 0.2,
    progress: bool = False,
    n_jobs: int = -1,
) -> tuple[CandidateRanker, TrainingReport]:
    """Train the candidate ranker on labelled plates."""
    X, y, _ = build_training_table(
        plates,
        stability_repeats=stability_repeats,
        seed=seed,
        progress=progress,
        n_jobs=n_jobs,
    )
    if len(X) < 20:
        raise ValueError("not enough training examples; generate more plates")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=15,
        l2_regularization=0.5,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)

    train_mae = float(np.mean(np.abs(model.predict(X_tr) - y_tr)))
    test_mae = float(np.mean(np.abs(model.predict(X_te) - y_te)))
    baseline_mae = float(np.mean(np.abs(y_te - y_tr.mean())))

    ranker = CandidateRanker(model=model, feature_names=list(FEATURE_NAMES))
    report = TrainingReport(
        n_plates=len(plates),
        n_examples=len(X),
        train_mae=train_mae,
        test_mae=test_mae,
        baseline_mae=baseline_mae,
        feature_importance=_permutation_importance(model, X_te, y_te, seed),
    )
    return ranker, report


def _permutation_importance(model, X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    """Drop in MAE when each feature is shuffled, as a plain readability aid."""
    rng = np.random.default_rng(seed)
    base = float(np.mean(np.abs(model.predict(X) - y)))
    out: dict[str, float] = {}
    for j, name in enumerate(FEATURE_NAMES):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        out[name] = float(np.mean(np.abs(model.predict(Xp) - y)) - base)
    return out


def save_report(report: TrainingReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
    return path
