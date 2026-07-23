"""
Pure-Python + NumPy implementation of the Langton's-Ant / turmite engine.

This is a faithful, readable port of AntFieldWidget's simulation core
(chunked sparse grid, per-cell visit/corner statistics, L/R rule strings).
It works everywhere with no build step, and is used automatically if the
compiled Cython extension (ant_engine_cy) hasn't been built yet.

For real workloads, build the Cython extension (see engine/setup.py) —
it exposes the exact same API and is a drop-in replacement.
"""
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import (
    CHUNK_AREA, CHUNK_SHIFT, CHUNK_SIZE, CORNER_LUT, DIRECTIONS,
    AntStatisticsSummary, chunk_index, compress_rules,
)

ENGINE_BACKEND = "python"  # overwritten to "cython" by the loader if compiled ext loads


class Chunk:
    __slots__ = ("states",)

    def __init__(self):
        self.states = np.zeros(CHUNK_AREA, dtype=np.uint32)


class StatChunk:
    __slots__ = ("visits", "corners", "first_visit_step", "last_visit_step")

    def __init__(self):
        self.visits = np.zeros(CHUNK_AREA, dtype=np.int64)
        self.corners = np.zeros((CHUNK_AREA, 4), dtype=np.int64)
        self.first_visit_step = np.full(CHUNK_AREA, -1, dtype=np.int64)
        self.last_visit_step = np.zeros(CHUNK_AREA, dtype=np.int64)


class AntEngine:
    """Sparse, chunked simulation of a generalized Langton's Ant ("turmite").

    Coordinates are arbitrary-precision Python ints (no overflow risk,
    unlike the original's fixed 64-bit qint64).
    """

    def __init__(self):
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.stat_chunks: Dict[Tuple[int, int], StatChunk] = {}

        self.rules: str = ""
        self.next_state_lut: List[int] = []
        self.direction_change_lut: List[int] = []

        self.statistics_enabled: bool = True
        self._lock = threading.Lock()

        # Ant / simulation state
        self.ant_x = 0
        self.ant_y = 0
        self.ant_dir = 0
        self.step_count = 0
        self.min_x = self.min_y = -50
        self.max_x = self.max_y = 50

        self.most_visited_cell = (0, 0)
        self.max_visits = 0
        self.unique_cells_count = 0
        self._sim_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Rules / lifecycle
    # ------------------------------------------------------------------ #
    def set_rules(self, rules_str: str):
        self.rules = rules_str.strip().upper()
        n = len(self.rules)
        self.next_state_lut = [(i + 1) % n for i in range(n)] if n else []
        self.direction_change_lut = [3 if c == "L" else 1 for c in self.rules]
        self.reset()

    def get_rules(self) -> str:
        return compress_rules(self.rules)

    def reset(self):
        with self._lock:
            self.chunks.clear()
            self.stat_chunks.clear()
            self.ant_x = 0
            self.ant_y = 0
            self.ant_dir = 0
            self.step_count = 0
            self.min_x = self.min_y = -50
            self.max_x = self.max_y = 50
            self.most_visited_cell = (0, 0)
            self.max_visits = 0
            self.unique_cells_count = 0
            self._sim_start = time.perf_counter()

    def reset_statistics(self):
        with self._lock:
            self.stat_chunks.clear()
            self.most_visited_cell = (0, 0)
            self.max_visits = 0
            self.unique_cells_count = 0
            self._sim_start = time.perf_counter()

    def set_statistics_enabled(self, enabled: bool):
        self.statistics_enabled = enabled
        if not enabled:
            self.reset_statistics()

    # ------------------------------------------------------------------ #
    # Core stepping
    # ------------------------------------------------------------------ #
    def _get_chunk(self, key):
        chunk = self.chunks.get(key)
        if chunk is None:
            chunk = Chunk()
            self.chunks[key] = chunk
        stat_chunk = None
        if self.statistics_enabled:
            stat_chunk = self.stat_chunks.get(key)
            if stat_chunk is None:
                stat_chunk = StatChunk()
                self.stat_chunks[key] = stat_chunk
        return chunk, stat_chunk

    def next_step(self, steps: int = 1):
        if not self.rules or steps <= 0:
            return

        n_states = len(self.next_state_lut)
        next_lut = self.next_state_lut
        dir_lut = self.direction_change_lut

        cx, cy = chunk_index(self.ant_x), chunk_index(self.ant_y)
        key = (cx, cy)
        chunk, stat_chunk = self._get_chunk(key)

        ant_x, ant_y, ant_dir = self.ant_x, self.ant_y, self.ant_dir
        step_count = self.step_count
        min_x, max_x, min_y, max_y = self.min_x, self.max_x, self.min_y, self.max_y
        most_visited_cell = self.most_visited_cell
        max_visits = self.max_visits
        unique_cells_count = self.unique_cells_count
        stats_on = self.statistics_enabled

        for _ in range(steps):
            new_cx, new_cy = chunk_index(ant_x), chunk_index(ant_y)
            if new_cx != cx or new_cy != cy:
                cx, cy = new_cx, new_cy
                key = (cx, cy)
                chunk, stat_chunk = self._get_chunk(key)

            lx = ant_x - cx * CHUNK_SIZE
            ly = ant_y - cy * CHUNK_SIZE
            idx = (ly << CHUNK_SHIFT) | lx

            states = chunk.states
            state = int(states[idx])
            old_dir = ant_dir
            if state < n_states:
                ant_dir = (ant_dir + dir_lut[state]) & 3
                states[idx] = next_lut[state]

            if stats_on and stat_chunk is not None:
                stat_chunk.visits[idx] += 1
                if stat_chunk.first_visit_step[idx] == -1:
                    stat_chunk.first_visit_step[idx] = step_count + 1
                    unique_cells_count += 1
                stat_chunk.last_visit_step[idx] = step_count + 1

                v = int(stat_chunk.visits[idx])
                if v > max_visits:
                    max_visits = v
                    most_visited_cell = (ant_x, ant_y)

                entry_side = (old_dir + 2) & 3
                corner_index = CORNER_LUT[(entry_side << 2) | ant_dir]
                if corner_index >= 0:
                    stat_chunk.corners[idx][corner_index] += 1

            dx, dy = DIRECTIONS[ant_dir]
            ant_x += dx
            ant_y += dy

            if ant_x < min_x:
                min_x = ant_x
            elif ant_x > max_x:
                max_x = ant_x
            if ant_y < min_y:
                min_y = ant_y
            elif ant_y > max_y:
                max_y = ant_y

            step_count += 1

        self.ant_x, self.ant_y, self.ant_dir = ant_x, ant_y, ant_dir
        self.step_count = step_count
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y
        if stats_on:
            with self._lock:
                self.most_visited_cell = most_visited_cell
                self.max_visits = max_visits
                self.unique_cells_count = unique_cells_count

    # ------------------------------------------------------------------ #
    # Area randomization
    # ------------------------------------------------------------------ #
    @staticmethod
    def estimate_randomize_area_bytes(radius: int) -> int:
        if radius < 0:
            return 0
        side = 2 * radius + 1
        chunks_per_side = (side + CHUNK_SIZE - 1) // CHUNK_SIZE
        total_chunks = chunks_per_side * chunks_per_side
        bytes_per_chunk = CHUNK_AREA * 4  # uint32 states
        return total_chunks * bytes_per_chunk

    def randomize_area(self, radius: int):
        if not self.rules or radius < 0:
            return
        state_count = len(self.rules)
        if state_count == 0:
            return

        start_x, end_x = self.ant_x - radius, self.ant_x + radius
        start_y, end_y = self.ant_y - radius, self.ant_y + radius
        start_cx, end_cx = chunk_index(start_x), chunk_index(end_x)
        start_cy, end_cy = chunk_index(start_y), chunk_index(end_y)

        rng = np.random.default_rng()

        for cy in range(start_cy, end_cy + 1):
            for cx in range(start_cx, end_cx + 1):
                key = (cx, cy)
                chunk = self.chunks.get(key)
                if chunk is None:
                    chunk = Chunk()
                    self.chunks[key] = chunk

                chunk_min_x, chunk_min_y = cx * CHUNK_SIZE, cy * CHUNK_SIZE
                local_start_x = max(start_x, chunk_min_x) - chunk_min_x
                local_end_x = min(end_x, chunk_min_x + CHUNK_SIZE - 1) - chunk_min_x
                local_start_y = max(start_y, chunk_min_y) - chunk_min_y
                local_end_y = min(end_y, chunk_min_y + CHUNK_SIZE - 1) - chunk_min_y

                grid = chunk.states.reshape(CHUNK_SIZE, CHUNK_SIZE)
                h = local_end_y - local_start_y + 1
                w = local_end_x - local_start_x + 1
                grid[local_start_y:local_end_y + 1, local_start_x:local_end_x + 1] = (
                    rng.integers(0, state_count, size=(h, w), dtype=np.uint32)
                )

        if start_x < self.min_x:
            self.min_x = start_x
        if end_x > self.max_x:
            self.max_x = end_x
        if start_y < self.min_y:
            self.min_y = start_y
        if end_y > self.max_y:
            self.max_y = end_y

    # ------------------------------------------------------------------ #
    # Statistics readback
    # ------------------------------------------------------------------ #
    def get_visit_count(self, x: int, y: int) -> int:
        if not self.statistics_enabled:
            return 0
        cx, cy = chunk_index(x), chunk_index(y)
        with self._lock:
            sc = self.stat_chunks.get((cx, cy))
            if sc is None:
                return 0
            lx, ly = x - cx * CHUNK_SIZE, y - cy * CHUNK_SIZE
            return int(sc.visits[(ly << CHUNK_SHIFT) | lx])

    def get_most_visited_cell(self) -> Tuple[int, int]:
        with self._lock:
            return self.most_visited_cell

    def get_statistics_summary(self) -> AntStatisticsSummary:
        with self._lock:
            summary = AntStatisticsSummary(
                total_cells_visited=self.step_count,
                max_visits_per_cell=self.max_visits,
                most_visited_cell=self.most_visited_cell,
                unique_cells_visited=self.unique_cells_count,
                simulation_time_ms=int((time.perf_counter() - self._sim_start) * 1000),
            )
            if summary.unique_cells_visited > 0:
                summary.average_visits = summary.total_cells_visited / summary.unique_cells_visited
            return summary

    def get_top_visited_cells(self, count: int) -> List[Tuple[Tuple[int, int], int]]:
        with self._lock:
            if not self.stat_chunks:
                return []
            all_cells = []
            for (kcx, kcy), sc in self.stat_chunks.items():
                nz = np.nonzero(sc.visits)[0]
                for i in nz:
                    i = int(i)
                    lx, ly = i % CHUNK_SIZE, i // CHUNK_SIZE
                    gx, gy = (kcx << CHUNK_SHIFT) + lx, (kcy << CHUNK_SHIFT) + ly
                    all_cells.append(((gx, gy), int(sc.visits[i])))
            all_cells.sort(key=lambda c: c[1], reverse=True)
            return all_cells[:count]

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #
    def save_state(self, filename: str) -> bool:
        import pickle
        try:
            with self._lock:
                data = {
                    "rules": self.rules,
                    "ant_x": self.ant_x, "ant_y": self.ant_y, "ant_dir": self.ant_dir,
                    "step_count": self.step_count,
                    "min_x": self.min_x, "max_x": self.max_x,
                    "min_y": self.min_y, "max_y": self.max_y,
                    "most_visited_cell": self.most_visited_cell,
                    "max_visits": self.max_visits,
                    "unique_cells_count": self.unique_cells_count,
                    "statistics_enabled": self.statistics_enabled,
                    "chunks": {k: v.states for k, v in self.chunks.items()},
                    "stat_chunks": {
                        k: (v.visits, v.corners, v.first_visit_step, v.last_visit_step)
                        for k, v in self.stat_chunks.items()
                    },
                }
            with open(filename, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except OSError:
            return False

    def load_state(self, filename: str) -> bool:
        import pickle
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        except (OSError, pickle.UnpicklingError):
            return False

        self.set_rules(data["rules"])  # triggers reset()
        with self._lock:
            self.ant_x, self.ant_y, self.ant_dir = data["ant_x"], data["ant_y"], data["ant_dir"]
            self.step_count = data["step_count"]
            self.min_x, self.max_x = data["min_x"], data["max_x"]
            self.min_y, self.max_y = data["min_y"], data["max_y"]
            self.most_visited_cell = data["most_visited_cell"]
            self.max_visits = data["max_visits"]
            self.unique_cells_count = data["unique_cells_count"]
            self.statistics_enabled = data["statistics_enabled"]

            self.chunks.clear()
            for k, states in data["chunks"].items():
                c = Chunk()
                c.states = states.astype(np.uint32, copy=True)
                self.chunks[k] = c

            self.stat_chunks.clear()
            for k, (visits, corners, fvs, lvs) in data["stat_chunks"].items():
                sc = StatChunk()
                sc.visits = visits.astype(np.int64, copy=True)
                sc.corners = corners.astype(np.int64, copy=True)
                sc.first_visit_step = fvs.astype(np.int64, copy=True)
                sc.last_visit_step = lvs.astype(np.int64, copy=True)
                self.stat_chunks[k] = sc
        return True
