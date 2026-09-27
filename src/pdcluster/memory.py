"""Persistent memory of perfect-score clusterings.

Project Discovery serves the same sample to many players, and plates repeat - so
once a plate has been solved *exactly right* there is no reason to ever search
again.  This module stores those solutions in SQLite, keyed by a fingerprint of
the point cloud, and recalls them for plates that are identical or merely very
similar.

Two recall paths:

* **exact** - the rounded coordinates hash to a stored key, so the stored labels
  are reused verbatim.
* **similar** - the density fingerprint is within `min_similarity`, e.g. the same
  sample re-sampled or lightly jittered.  Labels are transferred by nearest
  neighbour from the remembered cloud, then validated: if the transfer does not
  reproduce the remembered cluster count, the hit is rejected and the normal
  search runs instead.

By default *only* perfect-score solutions are written, which is what makes the
memory trustworthy enough to short-circuit the pipeline.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.spatial import cKDTree

from .types import NOISE, Plate

DEFAULT_MEMORY_PATH = Path.home() / ".pdcluster" / "memory.db"
FINGERPRINT_GRID = 16
FINGERPRINT_DIM = FINGERPRINT_GRID * FINGERPRINT_GRID

SCHEMA = """
CREATE TABLE IF NOT EXISTS solutions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_id      TEXT NOT NULL,
    content_hash  TEXT NOT NULL UNIQUE,
    fingerprint   BLOB NOT NULL,
    points        BLOB NOT NULL,
    labels        BLOB NOT NULL,
    n_points      INTEGER NOT NULL,
    n_clusters    INTEGER NOT NULL,
    algorithm     TEXT NOT NULL,
    params        TEXT NOT NULL,
    score         REAL NOT NULL,
    perfect       INTEGER NOT NULL,
    created_at    REAL NOT NULL,
    hits          INTEGER NOT NULL DEFAULT 0,
    last_hit_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_hash ON solutions(content_hash);
CREATE INDEX IF NOT EXISTS idx_perfect ON solutions(perfect);
"""


def fingerprint(plate: Plate) -> np.ndarray:
    """A translation/scale-invariant density signature of the point cloud.

    The plate is normalized into the unit square and binned into a coarse grid;
    the L2-normalized histogram is compared with cosine similarity.  Coarse enough
    to survive resampling and jitter, specific enough that different plates do not
    collide in practice.
    """
    pts = plate.normalized().points
    hist, _, _ = np.histogram2d(
        pts[:, 0],
        pts[:, 1],
        bins=FINGERPRINT_GRID,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    vec = hist.ravel().astype(np.float64)
    vec = np.sqrt(vec)  # compress dynamic range so one dense blob cannot dominate
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class MemoryHit:
    labels: np.ndarray
    similarity: float
    record_id: int
    kind: str  # "exact" | "similar"
    algorithm: str
    params: dict[str, Any]
    score: float
    n_clusters: int


@dataclass
class MemoryRecord:
    record_id: int
    plate_id: str
    content_hash: str
    n_points: int
    n_clusters: int
    algorithm: str
    params: dict[str, Any]
    score: float
    perfect: bool
    created_at: float
    hits: int


def _pack(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _unpack(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


class ClusterMemory:
    """SQLite-backed store of solved plates."""

    def __init__(
        self,
        path: str | Path | None = None,
        min_similarity: float = 0.985,
        store_only_perfect: bool = True,
        max_entries: int | None = 20000,
    ) -> None:
        self.path = Path(path) if path is not None else DEFAULT_MEMORY_PATH
        self.min_similarity = min_similarity
        self.store_only_perfect = store_only_perfect
        self.max_entries = max_entries
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # The app solves on a worker thread while the UI reads the memory on the
        # main one, and SQLite refuses a connection used from a thread other than
        # the one that made it.  One shared connection plus a lock is simpler than
        # a connection per thread, and keeps ":memory:" databases working (those
        # live inside a single connection and would otherwise be invisible).
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._cache: tuple[np.ndarray, list[int]] | None = None

    # ---------------------------------------------------------------- writing

    def remember(
        self,
        plate: Plate,
        labels: np.ndarray,
        score: float,
        algorithm: str,
        params: dict[str, Any] | None = None,
        perfect: bool | None = None,
    ) -> int | None:
        """Store a solution.  Returns the row id, or None when it was not stored."""
        with self._lock:
            return self._remember(plate, labels, score, algorithm, params, perfect)

    def _remember(
        self,
        plate: Plate,
        labels: np.ndarray,
        score: float,
        algorithm: str,
        params: dict[str, Any] | None = None,
        perfect: bool | None = None,
    ) -> int | None:
        labels = np.asarray(labels, dtype=np.int64)
        if perfect is None:
            perfect = score >= 0.999
        if self.store_only_perfect and not perfect:
            return None

        n_clusters = int(len(set(labels.tolist()) - {NOISE}))
        content_hash = plate.content_hash()
        row = self._conn.execute(
            "SELECT id, score FROM solutions WHERE content_hash = ?", (content_hash,)
        ).fetchone()

        if row is not None:
            # Keep whichever answer graded higher.
            if score <= row["score"]:
                return int(row["id"])
            self._conn.execute(
                "UPDATE solutions SET labels=?, score=?, perfect=?, algorithm=?, params=?,"
                " n_clusters=?, created_at=? WHERE id=?",
                (
                    _pack(labels),
                    float(score),
                    int(bool(perfect)),
                    algorithm,
                    json.dumps(params or {}, default=str),
                    n_clusters,
                    time.time(),
                    int(row["id"]),
                ),
            )
            self._conn.commit()
            self._cache = None
            return int(row["id"])

        cur = self._conn.execute(
            "INSERT INTO solutions (plate_id, content_hash, fingerprint, points, labels,"
            " n_points, n_clusters, algorithm, params, score, perfect, created_at, hits)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)",
            (
                plate.plate_id,
                content_hash,
                _pack(fingerprint(plate).astype(np.float32)),
                _pack(plate.points.astype(np.float32)),
                _pack(labels),
                int(len(plate.points)),
                n_clusters,
                algorithm,
                json.dumps(params or {}, default=str),
                float(score),
                int(bool(perfect)),
                time.time(),
            ),
        )
        self._conn.commit()
        self._cache = None
        self._enforce_cap()
        return int(cur.lastrowid)

    def _enforce_cap(self) -> None:
        if not self.max_entries:
            return
        (count,) = self._conn.execute("SELECT COUNT(*) FROM solutions").fetchone()
        if count <= self.max_entries:
            return
        surplus = count - self.max_entries
        # Evict the least useful first: never recalled, and oldest.
        self._conn.execute(
            "DELETE FROM solutions WHERE id IN ("
            " SELECT id FROM solutions ORDER BY hits ASC, created_at ASC LIMIT ?)",
            (surplus,),
        )
        self._conn.commit()
        self._cache = None

    # ---------------------------------------------------------------- reading

    def recall(self, plate: Plate, min_similarity: float | None = None) -> MemoryHit | None:
        """Look for a stored solution for this plate, exact first then similar."""
        with self._lock:
            return self._recall(plate, min_similarity)

    def _recall(self, plate: Plate, min_similarity: float | None = None) -> MemoryHit | None:
        threshold = self.min_similarity if min_similarity is None else min_similarity

        row = self._conn.execute(
            "SELECT * FROM solutions WHERE content_hash = ?", (plate.content_hash(),)
        ).fetchone()
        if row is not None:
            self._bump(int(row["id"]))
            return MemoryHit(
                labels=_unpack(row["labels"]),
                similarity=1.0,
                record_id=int(row["id"]),
                kind="exact",
                algorithm=row["algorithm"],
                params=json.loads(row["params"]),
                score=float(row["score"]),
                n_clusters=int(row["n_clusters"]),
            )

        matrix, ids = self._fingerprint_matrix()
        if not ids:
            return None

        query = fingerprint(plate)
        sims = matrix @ query
        best = int(np.argmax(sims))
        similarity = float(sims[best])
        if similarity < threshold:
            return None

        row = self._conn.execute("SELECT * FROM solutions WHERE id = ?", (ids[best],)).fetchone()
        if row is None:
            return None

        transferred = self._transfer_labels(
            _unpack(row["points"]).astype(np.float64), _unpack(row["labels"]), plate.points
        )
        if transferred is None:
            return None
        if int(len(set(transferred.tolist()) - {NOISE})) != int(row["n_clusters"]):
            # The transfer lost or invented a cluster: do not trust it.
            return None

        self._bump(int(row["id"]))
        return MemoryHit(
            labels=transferred,
            similarity=similarity,
            record_id=int(row["id"]),
            kind="similar",
            algorithm=row["algorithm"],
            params=json.loads(row["params"]),
            score=float(row["score"]),
            n_clusters=int(row["n_clusters"]),
        )

    @staticmethod
    def _transfer_labels(
        old_points: np.ndarray,
        old_labels: np.ndarray,
        new_points: np.ndarray,
        max_distance_quantile: float = 0.99,
    ) -> np.ndarray | None:
        """Carry remembered labels onto a new cloud by nearest neighbour.

        Points that land implausibly far from anything remembered are marked as
        noise rather than force-fitted into a cluster.
        """
        if len(old_points) == 0 or len(old_labels) != len(old_points):
            return None
        tree = cKDTree(old_points)
        dist, idx = tree.query(new_points, k=1)
        labels = old_labels[idx].astype(np.int64)
        if len(dist) > 10:
            cutoff = float(np.quantile(dist, max_distance_quantile)) * 3.0
            labels = np.where(dist > cutoff, NOISE, labels)
        return labels

    def _fingerprint_matrix(self) -> tuple[np.ndarray, list[int]]:
        if self._cache is not None:
            return self._cache
        rows = self._conn.execute(
            "SELECT id, fingerprint FROM solutions ORDER BY id"
        ).fetchall()
        if not rows:
            self._cache = (np.zeros((0, FINGERPRINT_DIM)), [])
            return self._cache
        matrix = np.vstack([_unpack(r["fingerprint"]).astype(np.float64) for r in rows])
        self._cache = (matrix, [int(r["id"]) for r in rows])
        return self._cache

    def _bump(self, record_id: int) -> None:
        self._conn.execute(
            "UPDATE solutions SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
            (time.time(), record_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------ maintenance

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM solutions").fetchone()
        return int(count)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return self._stats()

    def _stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) n, SUM(perfect) p, SUM(hits) h, AVG(n_clusters) k,"
            " AVG(score) s, MIN(created_at) first, MAX(created_at) last FROM solutions"
        ).fetchone()
        by_algo = {
            r["algorithm"]: int(r["c"])
            for r in self._conn.execute(
                "SELECT algorithm, COUNT(*) c FROM solutions GROUP BY algorithm ORDER BY c DESC"
            )
        }
        return {
            "path": str(self.path),
            "entries": int(row["n"] or 0),
            "perfect_entries": int(row["p"] or 0),
            "total_recalls": int(row["h"] or 0),
            "mean_clusters": round(float(row["k"] or 0.0), 3),
            "mean_score": round(float(row["s"] or 0.0), 4),
            "first_written": row["first"],
            "last_written": row["last"],
            "by_algorithm": by_algo,
        }

    def records(self, limit: int = 50, perfect_only: bool = False) -> Iterator[MemoryRecord]:
        sql = "SELECT * FROM solutions"
        if perfect_only:
            sql += " WHERE perfect = 1"
        sql += " ORDER BY hits DESC, created_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (limit,)).fetchall()
        for row in rows:
            yield MemoryRecord(
                record_id=int(row["id"]),
                plate_id=row["plate_id"],
                content_hash=row["content_hash"],
                n_points=int(row["n_points"]),
                n_clusters=int(row["n_clusters"]),
                algorithm=row["algorithm"],
                params=json.loads(row["params"]),
                score=float(row["score"]),
                perfect=bool(row["perfect"]),
                created_at=float(row["created_at"]),
                hits=int(row["hits"]),
            )

    def forget(self, record_id: int) -> bool:
        with self._lock:
            return self._forget(record_id)

    def _forget(self, record_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM solutions WHERE id = ?", (record_id,))
        self._conn.commit()
        self._cache = None
        return cur.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            return self._clear()

    def _clear(self) -> int:
        cur = self._conn.execute("DELETE FROM solutions")
        self._conn.commit()
        self._cache = None
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ClusterMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
