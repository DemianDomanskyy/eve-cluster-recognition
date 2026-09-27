"""End-to-end demo: solve a plate, prove the memory works, render figures.

Run it from the repository root::

    python scripts/demo.py --out out/

It needs no trained model - it falls back to the heuristic ranker - but it will
use `models/ranker.joblib` if you have trained one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pdcluster import ClusterMemory, ClusterRecognizer, make_plate
from pdcluster.plotting import cluster_table, format_cluster_table, plot_solution
from pdcluster.ranker import CandidateRanker
from pdcluster.types import Plate
from pdcluster.vision import points_from_image, render_plate_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out", help="directory for the figures")
    parser.add_argument("--model", default="models/ranker.joblib")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.12)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ranker = CandidateRanker()
    if Path(args.model).exists():
        ranker = CandidateRanker.load(args.model)
        print(f"using trained ranker: {args.model}")
    else:
        print("no trained model found - using the heuristic ranker")

    memory = ClusterMemory(out / "demo_memory.db")
    memory.clear()
    recognizer = ClusterRecognizer(ranker=ranker, memory=memory)

    plate = make_plate(
        seed=args.seed,
        n_clusters=args.clusters,
        n_points=900,
        noise_fraction=args.noise,
    )
    print(f"\nplate: {plate.n_points} points, {plate.n_true_clusters} true clusters")

    # 1. Cold solve: full candidate sweep ---------------------------------
    start = time.perf_counter()
    first = recognizer.solve(plate)
    cold = time.perf_counter() - start
    print(f"\n[1] cold solve       {cold * 1000:7.0f} ms   {first.summary()}")
    print(f"    considered {first.considered} candidate clusterings")
    print()
    print(format_cluster_table(cluster_table(plate, first.solution)))

    # 2. What the memory did with that answer -----------------------------
    graded_perfect = first.report is not None and first.report.is_perfect
    print()
    if graded_perfect:
        print("[2] graded perfect, so it was written to memory")
    else:
        print("[2] not perfect, so it was NOT written to memory - the store only keeps")
        print("    exact answers, which is what makes a recall safe to return unchecked.")
        print("    Seeding the verified answer by hand instead (recognizer.teach):")
        recognizer.teach(plate, plate.labels)
    print(f"    memory holds {len(memory)} solution(s)")

    # 3. Warm solve: the same plate again ---------------------------------
    start = time.perf_counter()
    second = recognizer.solve(plate)
    warm = time.perf_counter() - start
    print(f"\n[3] warm solve       {warm * 1000:7.0f} ms   {second.summary()}")
    if second.solution.source == "memory":
        print(f"    recalled from memory, {cold / max(warm, 1e-9):.0f}x faster than searching")
    else:
        print("    (no recall - the search ran again)")

    # 4. A near-duplicate plate: jittered, recognised by fingerprint ------
    rng = np.random.default_rng(0)
    jittered = Plate(points=plate.points + rng.normal(scale=0.0015, size=plate.points.shape))
    third = recognizer.solve(jittered)
    print(f"\n[4] jittered copy               {third.summary()}")
    if third.solution.source == "memory":
        print("    different coordinates, same populations - matched on the fingerprint")

    # 5. Straight from a screenshot ---------------------------------------
    shot = render_plate_image(plate, out / "screenshot.png", size=900, radius=2)
    from_pixels = points_from_image(shot)
    pixel_result = recognizer.solve(from_pixels, allow_memory=False)
    print(f"\n[5] from screenshot             {pixel_result.summary()}")
    print(f"    recovered {from_pixels.n_points} of {plate.n_points} markers from pixels")

    # 6. Figures -----------------------------------------------------------
    plot_solution(plate, second.solution, out / "solution_light.png", title="Solved plate")
    plot_solution(
        plate, second.solution, out / "solution_dark.png", title="Solved plate", dark=True
    )
    print(f"\nfigures written to {out}/")
    memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
