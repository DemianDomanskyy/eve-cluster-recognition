"""pdcluster - cluster recognition for Project Discovery style scatter plates.

Quick start::

    from pdcluster import ClusterRecognizer, make_plate

    plate = make_plate(seed=7)
    result = ClusterRecognizer().solve(plate)
    print(result.summary())
"""

from .clicks import ClickPlan, plan_clicks, simplify_to
from .loading import load_dataset, load_plate, save_dataset, save_plate
from .memory import ClusterMemory, MemoryHit, fingerprint
from .pipeline import ClusterRecognizer, EvalReport, SolveResult, evaluate
from .polygons import cluster_polygons, points_in_polygon
from .ranker import CandidateRanker, train_ranker
from .scoring import ScoreReport, is_perfect, score_clustering
from .synth import make_dataset, make_plate
from .types import NOISE, Candidate, Plate, Solution
from .vision import points_from_image, render_plate_image

__version__ = "0.1.0"

__all__ = [
    "NOISE",
    "Candidate",
    "ClickPlan",
    "ClusterMemory",
    "ClusterRecognizer",
    "CandidateRanker",
    "EvalReport",
    "MemoryHit",
    "Plate",
    "ScoreReport",
    "SolveResult",
    "Solution",
    "cluster_polygons",
    "evaluate",
    "fingerprint",
    "is_perfect",
    "load_dataset",
    "load_plate",
    "make_dataset",
    "make_plate",
    "plan_clicks",
    "simplify_to",
    "points_from_image",
    "points_in_polygon",
    "render_plate_image",
    "save_dataset",
    "save_plate",
    "score_clustering",
    "train_ranker",
    "__version__",
]
