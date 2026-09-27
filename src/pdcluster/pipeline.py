"""The end-to-end recognizer: remember, else search.

    plate -> memory lookup -------------------- hit -> reuse stored labels
                |
                miss
                v
          candidate sweep -> features -> ML ranker -> best labels -> polygons
                                                          |
                                      graded perfect? --> write to memory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .candidates import generate_candidates
from .features import extract_many, plate_features
from .memory import ClusterMemory, MemoryHit
from .polygons import cluster_polygons
from .ranker import CandidateRanker
from .scoring import ScoreReport, score_clustering
from .types import Candidate, Plate, Solution


@dataclass
class SolveResult:
    plate: Plate
    solution: Solution
    report: ScoreReport | None = None
    considered: int = 0
    runner_ups: list[tuple[str, float]] = None  # type: ignore[assignment]
    remembered_as: int | None = None

    def __post_init__(self) -> None:
        if self.runner_ups is None:
            self.runner_ups = []

    def summary(self) -> str:
        s = self.solution
        head = (
            f"{s.n_clusters} cluster(s) via {s.algorithm} "
            f"[{s.source}] predicted={s.predicted_score:.3f}"
            if s.predicted_score is not None
            else f"{s.n_clusters} cluster(s) via {s.algorithm} [{s.source}]"
        )
        if s.source == "memory" and s.memory_similarity is not None:
            head += f" similarity={s.memory_similarity:.4f}"
        if self.report is not None:
            head += f" | graded {self.report.summary()}"
        return head


class ClusterRecognizer:
    """Solve plates, and get better at it over time.

    The memory is checked first because a recalled perfect solution beats anything
    a fresh search can promise.  Whatever the search does produce is graded and
    written back whenever ground truth is available and the answer was exact.
    """

    def __init__(
        self,
        ranker: CandidateRanker | None = None,
        memory: ClusterMemory | None = None,
        use_memory: bool = True,
        write_memory: bool = True,
        stability_repeats: int = 2,
        max_k: int = 6,
        seed: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.ranker = ranker or CandidateRanker()
        self.memory = memory if memory is not None else (ClusterMemory() if use_memory else None)
        self.use_memory = use_memory and self.memory is not None
        self.write_memory = write_memory
        self.stability_repeats = stability_repeats
        self.max_k = max_k
        self.seed = seed
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ public

    @classmethod
    def from_paths(
        cls,
        model_path: str | Path | None = None,
        memory_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "ClusterRecognizer":
        ranker = CandidateRanker.load(model_path) if model_path else CandidateRanker()
        memory = ClusterMemory(memory_path) if kwargs.pop("use_memory", True) else None
        return cls(ranker=ranker, memory=memory, use_memory=memory is not None, **kwargs)

    def solve(
        self,
        plate: Plate,
        polygons: bool = True,
        grade: bool = True,
        allow_memory: bool | None = None,
    ) -> SolveResult:
        """Produce a solution for one plate."""
        allow_memory = self.use_memory if allow_memory is None else allow_memory

        if allow_memory and self.memory is not None:
            hit = self.memory.recall(plate)
            if hit is not None:
                return self._result_from_memory(plate, hit, polygons, grade)

        best, ranked = self._search(plate)
        solution = Solution(
            labels=best.labels,
            n_clusters=best.n_clusters,
            algorithm=best.algorithm,
            params=best.params,
            predicted_score=best.predicted_score,
            source="search",
        )
        if polygons:
            solution.polygons = cluster_polygons(plate.points, best.labels)

        report = None
        if grade and plate.labels is not None:
            report = score_clustering(plate.labels, best.labels)

        remembered = None
        if self.write_memory and self.memory is not None and report is not None:
            remembered = self.memory.remember(
                plate,
                best.labels,
                score=report.score,
                algorithm=best.algorithm,
                params=best.params,
                perfect=report.is_perfect,
            )

        return SolveResult(
            plate=plate,
            solution=solution,
            report=report,
            considered=len(ranked),
            runner_ups=[(c.describe(), float(c.predicted_score or 0.0)) for c in ranked[1:4]],
            remembered_as=remembered,
        )

    def solve_many(self, plates: list[Plate], **kwargs: Any) -> list[SolveResult]:
        return [self.solve(p, **kwargs) for p in plates]

    def teach(self, plate: Plate, labels: np.ndarray) -> int | None:
        """Store a known-correct answer directly, e.g. one verified by a human."""
        if self.memory is None:
            return None
        report = score_clustering(labels, labels)
        return self.memory.remember(
            plate, labels, score=report.score, algorithm="taught", params={}, perfect=True
        )

    # ----------------------------------------------------------------- private

    def _search(self, plate: Plate) -> tuple[Candidate, list[Candidate]]:
        candidates = generate_candidates(plate, max_k=self.max_k)
        pf = plate_features(plate, np.random.default_rng(self.seed))
        for cand, feats in zip(
            candidates,
            extract_many(
                plate,
                candidates,
                pf,
                stability_repeats=self.stability_repeats,
                seed=self.seed,
                n_jobs=self.n_jobs,
            ),
        ):
            cand.features = feats
        ranked = self.ranker.rank(candidates)
        return ranked[0], ranked

    def _result_from_memory(
        self, plate: Plate, hit: MemoryHit, polygons: bool, grade: bool
    ) -> SolveResult:
        solution = Solution(
            labels=hit.labels,
            n_clusters=hit.n_clusters,
            algorithm=hit.algorithm,
            params=hit.params,
            predicted_score=hit.score,
            source="memory",
            memory_similarity=hit.similarity,
        )
        if polygons:
            solution.polygons = cluster_polygons(plate.points, hit.labels)
        report = None
        if grade and plate.labels is not None:
            report = score_clustering(plate.labels, hit.labels)
        return SolveResult(
            plate=plate,
            solution=solution,
            report=report,
            considered=0,
            runner_ups=[],
            remembered_as=hit.record_id,
        )


@dataclass
class EvalReport:
    n_plates: int
    mean_score: float
    perfect_rate: float
    cluster_count_accuracy: float
    memory_hits: int
    mean_candidates: float
    per_algorithm: dict[str, int]

    def summary(self) -> str:
        algos = ", ".join(f"{k}={v}" for k, v in sorted(self.per_algorithm.items(), key=lambda kv: -kv[1]))
        return (
            f"plates={self.n_plates}  mean_score={self.mean_score:.4f}  "
            f"perfect={self.perfect_rate:.1%}  cluster_count_correct="
            f"{self.cluster_count_accuracy:.1%}  memory_hits={self.memory_hits}\n"
            f"winning algorithms: {algos}"
        )


def evaluate(
    recognizer: ClusterRecognizer,
    plates: list[Plate],
    polygons: bool = False,
    progress: bool = False,
) -> EvalReport:
    """Grade the recognizer over a labelled set of plates."""
    scores: list[float] = []
    perfect = 0
    counts_ok = 0
    mem_hits = 0
    considered: list[int] = []
    per_algo: dict[str, int] = {}

    for i, plate in enumerate(plates):
        result = recognizer.solve(plate, polygons=polygons, grade=True)
        if result.report is None:
            raise ValueError("evaluate() needs labelled plates")
        scores.append(result.report.score)
        perfect += int(result.report.is_perfect)
        counts_ok += int(result.report.cluster_count_correct)
        mem_hits += int(result.solution.source == "memory")
        considered.append(result.considered)
        key = result.solution.algorithm if result.solution.source == "search" else "memory"
        per_algo[key] = per_algo.get(key, 0) + 1
        if progress:
            print(f"  [{i + 1}/{len(plates)}] {result.summary()}", flush=True)

    n = max(len(plates), 1)
    return EvalReport(
        n_plates=len(plates),
        mean_score=float(np.mean(scores)) if scores else 0.0,
        perfect_rate=perfect / n,
        cluster_count_accuracy=counts_ok / n,
        memory_hits=mem_hits,
        mean_candidates=float(np.mean(considered)) if considered else 0.0,
        per_algorithm=per_algo,
    )
