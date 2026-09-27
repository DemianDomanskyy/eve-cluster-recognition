"""A desktop app for solving plates and drawing the loops by hand.

Two ways to work:

* **Auto** - solve the plate, let the planner decide how many sides each loop
  needs, and watch it place the clicks one at a time.
* **Draw** - do it yourself with the real mechanic: every click drops a vertex,
  an edge follows the one before it, and clicking the first vertex again closes
  the loop.  Nine sides maximum, exactly as in game.  Whatever you draw is
  scored against the plate immediately, so you can see how your loop did.

Launch it with `pdcluster gui`, or `python -m pdcluster.gui`.

This draws on its own canvas.  It does not send clicks to any other program.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import numpy as np

from .clicks import MAX_VERTICES, ClickPlan, plan_clicks, score_polygon
from .loading import load_plate
from .memory import ClusterMemory
from .pipeline import ClusterRecognizer
from .polygons import Polygon
from .ranker import CandidateRanker
from .scoring import score_clustering
from .synth import make_plate
from .types import NOISE, Plate

# The surface and the cluster hues are the validated dark-mode set from plotting.
BG = "#14141a"
PANEL = "#1c1c24"
PANEL_LIGHT = "#262631"
INK = "#f2f2f5"
INK_DIM = "#a0a0ae"
ACCENT = "#3987e5"
GOOD = "#199e70"
WARN = "#d95926"
GRID = "#2a2a34"
NOISE_COLOR = "#5a5a68"
CLUSTER_COLORS = ["#3987e5", "#d95926", "#199e70", "#9b8cf0", "#e0b341", "#43c6d8"]

SNAP_RADIUS = 14
"""How close a click has to be to the first vertex to close the loop, in pixels."""


def cluster_color(index: int) -> str:
    return CLUSTER_COLORS[index % len(CLUSTER_COLORS)]


class ClusterApp:
    """The main window."""

    def __init__(self, root: tk.Tk, model_path: str | None = None, memory_path: str | None = None):
        self.root = root
        self.root.title("pdcluster - Project Discovery cluster recognition")
        self.root.configure(bg=BG)
        # Sized to fit comfortably above a taskbar on a 1080p screen.
        self.root.geometry("1180x720")
        self.root.minsize(920, 600)

        self.plate: Plate | None = None
        self.labels: np.ndarray | None = None
        self.plans: list[ClickPlan] = []
        self.drawn: list[Polygon] = []
        self.current: list[list[float]] = []
        self.draw_mode = False
        self.show_truth = False
        self.hover: tuple[float, float] | None = None
        self.replay_job: str | None = None

        self.model_path = model_path
        self.memory_path = memory_path
        self._recognizer: ClusterRecognizer | None = None

        self._build()
        self.new_plate()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=PANEL, height=58)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="pdcluster",
            bg=PANEL,
            fg=INK,
            font=("Segoe UI Semibold", 16),
        ).pack(side="left", padx=(18, 8))
        tk.Label(
            header,
            text="find the populations, draw the loops",
            bg=PANEL,
            fg=INK_DIM,
            font=("Segoe UI", 10),
        ).pack(side="left")
        self.mode_badge = tk.Label(
            header, text="AUTO", bg=ACCENT, fg="#ffffff", font=("Segoe UI Semibold", 9), padx=10, pady=3
        )
        self.mode_badge.pack(side="right", padx=18)

        body = tk.Frame(self.root, bg=BG)
        body.pack(side="top", fill="both", expand=True)

        sidebar = tk.Frame(body, bg=PANEL, width=290)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(right, bg=BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="top", fill="both", expand=True, padx=12, pady=(12, 8))
        self.canvas.bind("<Configure>", lambda _e: self.redraw())
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Button-3>", lambda _e: self.undo_vertex())

        self.status = tk.Label(
            right,
            text="",
            bg=PANEL,
            fg=INK_DIM,
            anchor="w",
            font=("Consolas", 9),
            padx=12,
            pady=7,
        )
        self.status.pack(side="bottom", fill="x")

        # The report lives under the plot, not in the sidebar: the sidebar is a
        # fixed column of buttons, so a panel there collapses to nothing on a
        # short window, which is exactly where you most want to read the result.
        panel = tk.Frame(right, bg=PANEL, height=150)
        panel.pack(side="bottom", fill="x")
        panel.pack_propagate(False)
        tk.Label(
            panel, text="RESULT", bg=PANEL, fg=INK_DIM, font=("Segoe UI Semibold", 8), anchor="w"
        ).pack(fill="x", padx=14, pady=(8, 2))
        wrap = tk.Frame(panel, bg=PANEL)
        wrap.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        scroll = tk.Scrollbar(wrap, orient="vertical")
        scroll.pack(side="right", fill="y")
        self.result = tk.Text(
            wrap,
            bg=BG,
            fg=INK,
            insertbackground=INK,
            relief="flat",
            font=("Consolas", 8),
            wrap="word",
            padx=10,
            pady=6,
            yscrollcommand=scroll.set,
        )
        self.result.pack(side="left", fill="both", expand=True)
        scroll.configure(command=self.result.yview)
        self.result.configure(state="disabled")

    def _section(self, parent: tk.Widget, text: str) -> None:
        tk.Label(
            parent, text=text.upper(), bg=PANEL, fg=INK_DIM, font=("Segoe UI Semibold", 8), anchor="w"
        ).pack(fill="x", padx=16, pady=(16, 6))

    def _button(self, parent: tk.Widget, text: str, command, primary: bool = False) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=ACCENT if primary else PANEL_LIGHT,
            fg="#ffffff" if primary else INK,
            activebackground=ACCENT if primary else "#33333f",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=("Segoe UI", 9, "bold" if primary else "normal"),
            cursor="hand2",
            pady=7,
        )
        btn.pack(fill="x", padx=16, pady=3)
        return btn

    def _build_sidebar(self, bar: tk.Frame) -> None:
        self._section(bar, "Plate")
        row = tk.Frame(bar, bg=PANEL)
        row.pack(fill="x", padx=16, pady=3)
        tk.Label(row, text="clusters", bg=PANEL, fg=INK_DIM, font=("Segoe UI", 9)).pack(side="left")
        self.k_var = tk.StringVar(value="random")
        tk.OptionMenu(row, self.k_var, "random", "1", "2", "3", "4", "5").pack(side="right")
        self._button(bar, "New synthetic plate", self.new_plate)
        self._button(bar, "Open file / screenshot...", self.open_file)
        self.truth_btn = self._button(bar, "Show the true answer", self.toggle_truth)

        self._section(bar, "Automatic")
        self._button(bar, "Solve  +  plan clicks", self.solve, primary=True)
        self._button(bar, "Replay the clicks", self.replay)

        self._section(bar, "Draw it yourself")
        self.draw_btn = self._button(bar, "Start drawing", self.toggle_draw)
        self._button(bar, "Undo vertex  (right-click)", self.undo_vertex)
        self._button(bar, "Clear my loops", self.clear_drawn)

        self._section(bar, "Export")
        self._button(bar, "Save click plan (JSON)", self.export_clicks)


    # ------------------------------------------------------------- data access

    @property
    def recognizer(self) -> ClusterRecognizer:
        if self._recognizer is None:
            ranker = CandidateRanker()
            default_model = Path("models/ranker.joblib")
            path = self.model_path or (str(default_model) if default_model.exists() else None)
            if path:
                try:
                    ranker = CandidateRanker.load(path)
                    self.log(f"loaded ranker: {path}")
                except Exception as exc:  # pragma: no cover - defensive
                    self.log(f"could not load model ({exc}); using the heuristic")
            else:
                self.log("no trained model found - using the heuristic ranker")
            memory = ClusterMemory(self.memory_path) if self.memory_path else ClusterMemory()
            self._recognizer = ClusterRecognizer(ranker=ranker, memory=memory)
        return self._recognizer

    # --------------------------------------------------------------- reporting

    def log(self, text: str, clear: bool = False) -> None:
        self.result.configure(state="normal")
        if clear:
            self.result.delete("1.0", "end")
        self.result.insert("end", text + "\n")
        self.result.see("end")
        self.result.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    # ------------------------------------------------------------------ actions

    def new_plate(self) -> None:
        choice = self.k_var.get() if hasattr(self, "k_var") else "random"
        k = None if choice == "random" else int(choice)
        seed = int(np.random.default_rng().integers(0, 10**6))
        self.plate = make_plate(seed=seed, n_clusters=k, n_points=900)
        self.reset_state()
        self.log(
            f"new plate: {self.plate.n_points} points, "
            f"{self.plate.n_true_clusters} true cluster(s), seed {seed}",
            clear=True,
        )
        self.set_status("Press 'Solve + plan clicks', or start drawing loops yourself.")
        self.redraw()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open a plate",
            filetypes=[
                ("All supported", "*.npz *.npy *.csv *.tsv *.json *.png *.jpg *.jpeg *.bmp"),
                ("Data", "*.npz *.npy *.csv *.tsv *.json"),
                ("Screenshot", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.plate = load_plate(path)
        except Exception as exc:
            messagebox.showerror("Could not open", str(exc))
            return
        self.reset_state()
        self.log(f"loaded {Path(path).name}: {self.plate.n_points} points", clear=True)
        if self.plate.n_points == 0:
            self.log("no points found - if this is a screenshot, crop it to the plot area")
        self.set_status(f"Loaded {Path(path).name}")
        self.redraw()

    def reset_state(self) -> None:
        self.labels = None
        self.plans = []
        self.drawn = []
        self.current = []
        self.cancel_replay()

    def toggle_truth(self) -> None:
        """Colour the points by the reference answer instead of the solver's."""
        if self.plate is None or self.plate.labels is None:
            self.set_status("this plate has no reference answer to show")
            return
        self.show_truth = not self.show_truth
        self.truth_btn.configure(
            text="Show what was found" if self.show_truth else "Show the true answer"
        )
        self.set_status(
            "showing the true answer" if self.show_truth else "showing what the solver found"
        )
        self.redraw()

    def solve(self) -> None:
        if self.plate is None or self.plate.n_points == 0:
            return
        self.cancel_replay()
        self.set_status("solving - sweeping candidate clusterings...")
        self.root.update_idletasks()

        result = self.recognizer.solve(self.plate, polygons=True)
        self.labels = result.solution.labels
        lo, hi = self.plate.points.min(axis=0), self.plate.points.max(axis=0)
        self.plans = plan_clicks(
            self.plate,
            result.solution,
            bounds=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
        )

        self.log(f"\n{result.solution.n_clusters} cluster(s) via {result.solution.algorithm}"
                 f" [{result.solution.source}]", clear=True)
        if result.considered:
            self.log(f"considered {result.considered} candidate clusterings")
        for plan in self.plans:
            self.log(plan.describe())
            by_k = " ".join(f"{k}:{v:.2f}" for k, v in sorted(plan.considered.items()))
            self.log(f"   score by side count -> {by_k}")
        if result.report is not None:
            self.log(f"\ngraded: {result.report.summary()}")

        total = sum(p.n_clicks for p in self.plans)
        self.set_status(
            f"{len(self.plans)} loop(s), {total} clicks total. 'Replay the clicks' to watch it draw."
        )
        self.redraw()

    def replay(self) -> None:
        if not self.plans:
            self.solve()
        if not self.plans:
            return
        self.cancel_replay()
        sequence: list[tuple[int, int]] = []
        for i, plan in enumerate(self.plans):
            for j in range(plan.n_clicks):
                sequence.append((i, j + 1))
        self._replay_step(sequence, 0)

    def _replay_step(self, sequence: list[tuple[int, int]], index: int) -> None:
        if index >= len(sequence):
            self.replay_job = None
            self.set_status("replay finished - every loop closed on its first vertex")
            self.redraw()
            return
        self.replay_progress = sequence[index]
        cluster, clicks = sequence[index]
        plan = self.plans[cluster]
        closing = clicks == plan.n_clicks
        self.set_status(
            f"loop {cluster + 1}/{len(self.plans)} - click {clicks}/{plan.n_clicks}"
            + ("  (closing on the first vertex)" if closing else "")
        )
        self.redraw()
        self.replay_job = self.root.after(
            260 if not closing else 500, lambda: self._replay_step(sequence, index + 1)
        )

    def cancel_replay(self) -> None:
        if self.replay_job is not None:
            self.root.after_cancel(self.replay_job)
            self.replay_job = None
        self.replay_progress = None

    def toggle_draw(self) -> None:
        self.draw_mode = not self.draw_mode
        self.current = []
        self.cancel_replay()
        self.draw_btn.configure(text="Stop drawing" if self.draw_mode else "Start drawing")
        self.mode_badge.configure(
            text="DRAW" if self.draw_mode else "AUTO", bg=WARN if self.draw_mode else ACCENT
        )
        if self.draw_mode:
            self.set_status(
                f"Click to place a vertex. Max {MAX_VERTICES} sides. "
                "Click the first vertex again to close the loop."
            )
        else:
            self.set_status("Drawing off.")
        self.redraw()

    def undo_vertex(self) -> None:
        if self.current:
            self.current.pop()
            self.redraw()
            self.set_status(f"{len(self.current)} vertex/vertices placed")

    def clear_drawn(self) -> None:
        self.drawn = []
        self.current = []
        self.redraw()
        self.set_status("cleared the loops you drew")

    def export_clicks(self) -> None:
        if not self.plans and not self.drawn:
            messagebox.showinfo("Nothing to export", "Solve a plate or draw a loop first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="clicks.json"
        )
        if not path:
            return
        payload: dict[str, Any] = {
            "plate_id": self.plate.plate_id if self.plate else None,
            "n_points": self.plate.n_points if self.plate else 0,
            "coordinate_space": "plate coordinates, origin bottom-left",
            "planned": [p.to_dict() for p in self.plans],
            "drawn_by_hand": [
                {"sides": len(poly), "clicks": [list(v) for v in poly] + [list(poly[0])]}
                for poly in self.drawn
                if poly
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.log(f"\nexported -> {path}")
        self.set_status(f"click plan written to {Path(path).name}")

    # ------------------------------------------------------------ click handling

    def on_motion(self, event: tk.Event) -> None:
        if self.draw_mode and self.current:
            self.hover = (event.x, event.y)
            self.redraw()

    def on_click(self, event: tk.Event) -> None:
        if not self.draw_mode or self.plate is None:
            return

        # Closing the loop: click the first vertex again.
        if len(self.current) >= 3:
            first = self.to_canvas(*self.current[0])
            if (event.x - first[0]) ** 2 + (event.y - first[1]) ** 2 <= SNAP_RADIUS**2:
                self.close_loop()
                return

        if len(self.current) >= MAX_VERTICES:
            self.set_status(
                f"{MAX_VERTICES} vertices is the limit - click the first vertex to close the loop."
            )
            return

        self.current.append(list(self.to_data(event.x, event.y)))
        remaining = MAX_VERTICES - len(self.current)
        hint = (
            "click the first vertex to close"
            if len(self.current) >= 3
            else f"{3 - len(self.current)} more before you can close"
        )
        self.set_status(f"vertex {len(self.current)}/{MAX_VERTICES} - {remaining} left, {hint}")
        self.redraw()

    def close_loop(self) -> None:
        polygon = [list(v) for v in self.current]
        self.drawn.append(polygon)
        self.current = []
        self.hover = None
        self.report_drawn(polygon)
        self.redraw()

    def report_drawn(self, polygon: Polygon) -> None:
        """Grade a hand-drawn loop against whatever the plate actually contains."""
        if self.plate is None:
            return
        truth = self.plate.labels if self.plate.labels is not None else self.labels
        index = len(self.drawn)
        if truth is None:
            self.log(f"\nloop {index}: {len(polygon)} sides (no reference to grade against)")
            self.set_status(f"loop {index} closed - {len(polygon)} sides")
            return

        ids = sorted(set(np.asarray(truth).tolist()) - {NOISE})
        best, best_f1 = None, -1.0
        for cluster in ids:
            own = self.plate.points[truth == cluster]
            others = self.plate.points[truth != cluster]
            capture, purity, f1 = score_polygon(polygon, own, others)
            if f1 > best_f1:
                best, best_f1 = (cluster, capture, purity, f1), f1

        if best is None:
            self.log(f"\nloop {index}: {len(polygon)} sides, matched no cluster")
            return
        cluster, capture, purity, f1 = best
        self.log(
            f"\nloop {index}: {len(polygon)} sides, {len(polygon) + 1} clicks\n"
            f"   best match: cluster {cluster + 1}\n"
            f"   capture {capture:.1%}  purity {purity:.1%}  score {f1:.3f}"
        )
        plan = next((p for p in self.plans if p.cluster == cluster), None)
        if plan is not None:
            delta = f1 - plan.score
            verdict = "better than" if delta > 0.005 else ("on par with" if abs(delta) <= 0.005 else "below")
            self.log(f"   {verdict} the planner ({plan.score:.3f} with {plan.n_sides} sides)")
        self.set_status(f"loop {index} closed - score {f1:.3f} against cluster {cluster + 1}")

    # ------------------------------------------------------------------ drawing

    def geometry(self) -> tuple[float, float, float]:
        """Return (origin_x, origin_y, scale) mapping plate space to the canvas."""
        w = max(self.canvas.winfo_width(), 50)
        h = max(self.canvas.winfo_height(), 50)
        pad = 28
        size = max(min(w, h) - 2 * pad, 10)
        return (w - size) / 2, (h - size) / 2, size

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        ox, oy, size = self.geometry()
        return ox + x * size, oy + (1.0 - y) * size

    def to_data(self, px: float, py: float) -> tuple[float, float]:
        ox, oy, size = self.geometry()
        return (px - ox) / size, 1.0 - (py - oy) / size

    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        if self.plate is None:
            return

        ox, oy, size = self.geometry()
        c.create_rectangle(ox, oy, ox + size, oy + size, outline=GRID, width=1)
        for i in range(1, 4):
            t = i / 4
            c.create_line(ox + t * size, oy, ox + t * size, oy + size, fill=GRID)
            c.create_line(ox, oy + t * size, ox + size, oy + t * size, fill=GRID)

        self._draw_points()
        self._draw_planned()
        self._draw_polygons(self.drawn, WARN, "your loop")
        self._draw_current()

    def _draw_points(self) -> None:
        pts = self.plate.points
        if self.show_truth and self.plate.labels is not None:
            labels = self.plate.labels
        else:
            labels = self.labels if self.labels is not None else self.plate.labels
        r = 2
        if labels is None:
            for x, y in pts:
                px, py = self.to_canvas(x, y)
                self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=INK_DIM, outline="")
            return
        ids = sorted(set(np.asarray(labels).tolist()) - {NOISE})
        for x, y in pts[labels == NOISE]:
            px, py = self.to_canvas(x, y)
            self.canvas.create_oval(px - 1.5, py - 1.5, px + 1.5, py + 1.5, fill=NOISE_COLOR, outline="")
        for i, cluster in enumerate(ids):
            color = cluster_color(i)
            for x, y in pts[labels == cluster]:
                px, py = self.to_canvas(x, y)
                self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="")

    def _draw_planned(self) -> None:
        """Draw planned loops, honouring how far a replay has progressed."""
        for i, plan in enumerate(self.plans):
            color = cluster_color(i)
            shown = plan.n_clicks
            progress = getattr(self, "replay_progress", None)
            if progress is not None:
                active, clicks = progress
                if i > active:
                    continue
                shown = clicks if i == active else plan.n_clicks

            path = plan.clicks[:shown]
            if len(path) >= 2:
                flat: list[float] = []
                for vx, vy in path:
                    flat.extend(self.to_canvas(vx, vy))
                self.canvas.create_line(*flat, fill=color, width=2, joinstyle="round")
            for j, (vx, vy) in enumerate(path[: plan.n_sides]):
                px, py = self.to_canvas(vx, vy)
                first = j == 0
                self.canvas.create_oval(
                    px - 5, py - 5, px + 5, py + 5,
                    fill=BG if first else color,
                    outline=color,
                    width=2,
                )
            if path:
                lx, ly = self.to_canvas(*plan.vertices[0])
                self.canvas.create_text(
                    lx, ly - 16, text=str(i + 1), fill=color, font=("Segoe UI Semibold", 10)
                )

    def _draw_polygons(self, polygons: list[Polygon], color: str, _label: str) -> None:
        for poly in polygons:
            if len(poly) < 2:
                continue
            flat: list[float] = []
            for vx, vy in list(poly) + [poly[0]]:
                flat.extend(self.to_canvas(vx, vy))
            self.canvas.create_line(*flat, fill=color, width=2, joinstyle="round")
            for vx, vy in poly:
                px, py = self.to_canvas(vx, vy)
                self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill=color, outline="")

    def _draw_current(self) -> None:
        """The loop in progress: placed vertices, edges, and the rubber band."""
        if not self.current:
            return
        flat: list[float] = []
        for vx, vy in self.current:
            flat.extend(self.to_canvas(vx, vy))
        if len(self.current) >= 2:
            self.canvas.create_line(*flat, fill=WARN, width=2, joinstyle="round")

        if self.hover is not None:
            last = self.to_canvas(*self.current[-1])
            self.canvas.create_line(
                last[0], last[1], self.hover[0], self.hover[1], fill=WARN, width=1, dash=(4, 3)
            )

        for i, (vx, vy) in enumerate(self.current):
            px, py = self.to_canvas(vx, vy)
            if i == 0:
                # The start vertex is the target that closes the loop.
                ready = len(self.current) >= 3
                self.canvas.create_oval(
                    px - SNAP_RADIUS, py - SNAP_RADIUS, px + SNAP_RADIUS, py + SNAP_RADIUS,
                    outline=GOOD if ready else GRID, width=2, dash=(3, 3),
                )
                self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill=BG, outline=WARN, width=2)
            else:
                self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill=WARN, outline="")

        counter = f"{len(self.current)}/{MAX_VERTICES} vertices"
        ox, oy, size = self.geometry()
        self.canvas.create_text(
            ox + size - 6, oy + 10, text=counter, fill=INK_DIM, anchor="e", font=("Consolas", 9)
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the pdcluster desktop app.")
    parser.add_argument("--model", help="path to a trained ranker (.joblib)")
    parser.add_argument("--memory", help="path to the solution memory database")
    args = parser.parse_args(argv)

    root = tk.Tk()
    ClusterApp(root, model_path=args.model, memory_path=args.memory)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
