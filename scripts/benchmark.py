"""Benchmark the ranker against its own ceiling.

Three numbers on the same plates:

* **oracle**    - the best candidate the sweep produced, chosen with the answer
                  key in hand.  This is the ceiling: no ranker can beat it
                  without a better candidate pool.
* **trained**   - the gradient-boosted ranker's pick.
* **heuristic** - the untrained cold-start fallback's pick.

The gap between heuristic and trained is what the learning bought.  The gap
between trained and oracle is what is left to win by ranking better; if that gap
is small, further gains have to come from generating better candidates instead.

    python scripts/benchmark.py --dataset data/test --model models/ranker.joblib
"""

from __future__ import annotations

import argparse

import numpy as np

from pdcluster.candidates import generate_candidates
from pdcluster.features import candidate_features, plate_features
from pdcluster.loading import load_dataset
from pdcluster.ranker import CandidateRanker
from pdcluster.scoring import score_clustering
from pdcluster.synth import make_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="directory of labelled plates")
    parser.add_argument("--plates", type=int, default=40)
    parser.add_argument("--model", default="models/ranker.joblib")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--stability-repeats", type=int, default=2)
    args = parser.parse_args()

    plates = load_dataset(args.dataset) if args.dataset else make_dataset(args.plates, args.seed)
    trained = CandidateRanker.load(args.model)
    heuristic = CandidateRanker()
    rng = np.random.default_rng(0)

    rows: dict[str, list[float]] = {"oracle": [], "trained": [], "heuristic": []}
    perfect: dict[str, int] = {"oracle": 0, "trained": 0, "heuristic": 0}
    counts: dict[str, int] = {"oracle": 0, "trained": 0, "heuristic": 0}

    for i, plate in enumerate(plates):
        pf = plate_features(plate, rng)
        cands = generate_candidates(plate)
        for cand in cands:
            cand.features = candidate_features(
                plate, cand, pf, stability_repeats=args.stability_repeats, rng=rng
            )
            cand.true_score = score_clustering(plate.labels, cand.labels).score

        picks = {
            "oracle": max(cands, key=lambda c: c.true_score or 0.0),
            "trained": max(cands, key=lambda c: trained.predict(c.features)),
            "heuristic": max(cands, key=lambda c: heuristic.predict(c.features)),
        }
        for name, cand in picks.items():
            report = score_clustering(plate.labels, cand.labels)
            rows[name].append(report.score)
            perfect[name] += int(report.is_perfect)
            counts[name] += int(report.cluster_count_correct)

        print(f"  [{i + 1}/{len(plates)}] " + "  ".join(
            f"{n}={score_clustering(plate.labels, c.labels).score:.3f}" for n, c in picks.items()
        ), flush=True)

    n = len(plates)
    print(f"\n{'':<12}{'mean score':>12}{'perfect':>10}{'k correct':>12}")
    for name in ("heuristic", "trained", "oracle"):
        print(
            f"{name:<12}{np.mean(rows[name]):>12.4f}"
            f"{perfect[name] / n:>9.1%}{counts[name] / n:>12.1%}"
        )
    gap_learned = np.mean(rows["trained"]) - np.mean(rows["heuristic"])
    gap_left = np.mean(rows["oracle"]) - np.mean(rows["trained"])
    print(f"\nlearning gained {gap_learned:+.4f}; {gap_left:.4f} of ranking headroom remains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
