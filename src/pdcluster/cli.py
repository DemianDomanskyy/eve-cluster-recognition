"""Command line interface: `pdcluster <command>`.

    pdcluster synth   --out data/train --plates 150
    pdcluster train   --dataset data/train --model models/ranker.joblib
    pdcluster solve   --input shot.png --model models/ranker.joblib --plot out.png
    pdcluster eval    --dataset data/test --model models/ranker.joblib --repeat
    pdcluster memory  stats
    pdcluster gui
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .loading import load_dataset, load_plate, save_dataset
from .memory import ClusterMemory
from .pipeline import ClusterRecognizer, evaluate
from .plotting import cluster_table, format_cluster_table, plot_solution
from .ranker import CandidateRanker, save_report, train_ranker
from .synth import make_dataset

DEFAULT_MODEL = Path("models/ranker.joblib")


def _load_ranker(path: str | None) -> CandidateRanker:
    if path:
        return CandidateRanker.load(path)
    if DEFAULT_MODEL.exists():
        print(f"[i] using {DEFAULT_MODEL}", file=sys.stderr)
        return CandidateRanker.load(DEFAULT_MODEL)
    print("[!] no trained model; falling back to the heuristic ranker", file=sys.stderr)
    return CandidateRanker()


# --------------------------------------------------------------------- commands


def cmd_synth(args: argparse.Namespace) -> int:
    plates = make_dataset(n_plates=args.plates, seed=args.seed)
    save_dataset(plates, args.out)
    counts = np.bincount([p.n_true_clusters or 0 for p in plates], minlength=7)
    print(f"wrote {len(plates)} plates to {args.out}")
    print("cluster-count distribution: " + ", ".join(f"{k}:{v}" for k, v in enumerate(counts) if v))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    plates = load_dataset(args.dataset) if args.dataset else make_dataset(args.plates, args.seed)
    if not args.dataset:
        print(f"[i] no --dataset given; generated {len(plates)} synthetic plates")
    print(f"training on {len(plates)} plates (this enumerates and grades every candidate)...")
    ranker, report = train_ranker(
        plates,
        stability_repeats=args.stability_repeats,
        seed=args.seed,
        progress=args.progress,
    )
    out = ranker.save(args.model)
    print(report.summary())
    print(f"model -> {out}")
    if args.report:
        print(f"report -> {save_report(report, args.report)}")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    plate = load_plate(args.input)
    if plate.n_points == 0:
        print("no points found in input", file=sys.stderr)
        return 2

    recognizer = ClusterRecognizer(
        ranker=_load_ranker(args.model),
        memory=None if args.no_memory else ClusterMemory(args.memory),
        use_memory=not args.no_memory,
        write_memory=not args.no_memory,
        stability_repeats=args.stability_repeats,
        max_k=args.max_k,
    )
    result = recognizer.solve(plate, polygons=True)
    print(f"{args.input}: {result.summary()}")
    print(f"points={plate.n_points} candidates_considered={result.considered}")
    if result.runner_ups:
        print("runner-ups: " + ", ".join(f"{d} ({s:.3f})" for d, s in result.runner_ups))
    print()
    print(format_cluster_table(cluster_table(plate, result.solution)))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(result.solution.to_json(), encoding="utf-8")
        print(f"\nsolution -> {args.out}")
    if args.plot:
        print(f"figure   -> {plot_solution(plate, result.solution, args.plot, dark=args.dark)}")
    if args.clicks:
        from .clicks import plan_clicks

        plans = plan_clicks(plate, result.solution, max_vertices=args.max_vertices)
        print()
        for plan in plans:
            print("  " + plan.describe())
        Path(args.clicks).parent.mkdir(parents=True, exist_ok=True)
        Path(args.clicks).write_text(
            json.dumps([p.to_dict() for p in plans], indent=2), encoding="utf-8"
        )
        print(f"clicks   -> {args.clicks}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    plates = load_dataset(args.dataset) if args.dataset else make_dataset(args.plates, args.seed)
    if not args.dataset:
        print(f"[i] no --dataset given; generated {len(plates)} synthetic plates")

    memory = ClusterMemory(args.memory) if args.memory else ClusterMemory(":memory:")
    recognizer = ClusterRecognizer(
        ranker=_load_ranker(args.model),
        memory=memory,
        use_memory=True,
        write_memory=True,
        stability_repeats=args.stability_repeats,
        max_k=args.max_k,
    )

    print("=== pass 1: cold memory ===")
    first = evaluate(recognizer, plates, progress=args.progress)
    print(first.summary())
    print(f"memory now holds {len(memory)} perfect solution(s)")

    if args.repeat:
        print("\n=== pass 2: same plates again, memory warm ===")
        second = evaluate(recognizer, plates, progress=args.progress)
        print(second.summary())
        recalled = second.memory_hits
        print(
            f"recalled {recalled}/{len(plates)} plates from memory "
            f"({recalled / max(len(plates), 1):.1%}), "
            f"mean score {first.mean_score:.4f} -> {second.mean_score:.4f}"
        )
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import main as gui_main

    argv: list[str] = []
    if args.model:
        argv += ["--model", args.model]
    if args.memory:
        argv += ["--memory", args.memory]
    return gui_main(argv)


def cmd_memory(args: argparse.Namespace) -> int:
    memory = ClusterMemory(args.memory)
    if args.action == "stats":
        print(json.dumps(memory.stats(), indent=2))
    elif args.action == "list":
        for rec in memory.records(limit=args.limit, perfect_only=args.perfect_only):
            print(
                f"#{rec.record_id:<6} {rec.content_hash}  k={rec.n_clusters} "
                f"n={rec.n_points:<5} score={rec.score:.4f} hits={rec.hits:<4} {rec.algorithm}"
            )
    elif args.action == "forget":
        if args.id is None:
            print("--id is required for forget", file=sys.stderr)
            return 2
        print("forgotten" if memory.forget(args.id) else "no such record")
    elif args.action == "clear":
        print(f"removed {memory.clear()} record(s)")
    return 0


# ----------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdcluster",
        description="Cluster recognition for Project Discovery style scatter plates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common_model = {"--model": "path to a trained ranker (.joblib)"}

    p = sub.add_parser("synth", help="generate a labelled synthetic dataset")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--plates", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("train", help="train the candidate ranker")
    p.add_argument("--dataset", help="directory written by `synth`; synthesized if omitted")
    p.add_argument("--plates", type=int, default=150, help="plates to synthesize if no --dataset")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--report", help="write the training report as JSON")
    p.add_argument("--stability-repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("solve", help="solve one plate (npz/csv/json or a screenshot)")
    p.add_argument("--input", required=True)
    p.add_argument("--model", help=common_model["--model"])
    p.add_argument("--memory", help="memory database path")
    p.add_argument("--no-memory", action="store_true", help="ignore and do not write memory")
    p.add_argument("--out", help="write the solution as JSON")
    p.add_argument("--plot", help="write a PNG figure")
    p.add_argument("--dark", action="store_true", help="render the figure on a dark surface")
    p.add_argument("--clicks", help="write the click plan for each cluster as JSON")
    p.add_argument("--max-vertices", type=int, default=9, help="polygon side limit (default 9)")
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--stability-repeats", type=int, default=2)
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("eval", help="grade the recognizer on labelled plates")
    p.add_argument("--dataset")
    p.add_argument("--plates", type=int, default=40)
    p.add_argument("--model", help=common_model["--model"])
    p.add_argument("--memory", help="memory database path (a temporary one by default)")
    p.add_argument("--repeat", action="store_true", help="second pass to show memory recall")
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--stability-repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--progress", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("gui", help="launch the desktop app")
    p.add_argument("--model", help=common_model["--model"])
    p.add_argument("--memory", help="memory database path")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("memory", help="inspect the solution memory")
    p.add_argument("action", choices=["stats", "list", "forget", "clear"])
    p.add_argument("--memory", help="memory database path")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--perfect-only", action="store_true")
    p.add_argument("--id", type=int)
    p.set_defaults(func=cmd_memory)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
