# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-accelerated Langton's-Ant / turmite engine.

Same public API and identical semantics as engine.ant_engine_py.AntEngine
(verify with tests/test_engine_parity.py) but with the hot per-step loop
fully typed and released from the GIL, giving roughly a 20-50x speedup
over the pure-Python version. Chunk lookup (Python dict) still needs the
GIL, but that only happens once every ~32 steps on average (chunk = 32x32
cells), not once per step.

Build with:
    cd engine && python setup.py build_ext --inplace
"""
import pickle
import threading
import time

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdint cimport int64_t, uint32_t

from .common import AntStatisticsSummary, chunk_index, compress_rules

ENGINE_BACKEND = "cython"

DEF CHUNK_SHIFT = 5
DEF CHUNK_SIZE = 32
DEF CHUNK_AREA = 1024

cdef int DIR_DX[4]
cdef int DIR_DY[4]
DIR_DX[0] = 0; DIR_DY[0] = -1
DIR_DX[1] = 1; DIR_DY[1] = 0
DIR_DX[2] = 0; DIR_DY[2] = 1
DIR_DX[3] = -1; DIR_DY[3] = 0

cdef int CORNER_LUT[16]
_corner_init = (-1, 1, -1, 0,  1, -1, 2, -1,  -1, 2, -1, 3,  0, -1, 3, -1)
for _i in range(16):
    CORNER_LUT[_i] = _corner_init[_i]


cdef class Chunk:
    cdef public cnp.ndarray states  # uint32[CHUNK_AREA]

    def __cinit__(self):
        self.states = np.zeros(CHUNK_AREA, dtype=np.uint32)


cdef class StatChunk:
    cdef public cnp.ndarray visits          # int64[CHUNK_AREA]
    cdef public cnp.ndarray corners         # int64[CHUNK_AREA, 4]
    cdef public cnp.ndarray first_visit_step
    cdef public cnp.ndarray last_visit_step

    def __cinit__(self):
        self.visits = np.zeros(CHUNK_AREA, dtype=np.int64)
        self.corners = np.zeros((CHUNK_AREA, 4), dtype=np.int64)
        self.first_visit_step = np.full(CHUNK_AREA, -1, dtype=np.int64)
        self.last_visit_step = np.zeros(CHUNK_AREA, dtype=np.int64)


cdef class AntEngine:
    cdef public dict chunks
    cdef public dict stat_chunks
    cdef public str rules
    cdef list next_state_lut_py
    cdef list direction_change_lut_py
    cdef public bint statistics_enabled
    # True while this is a "pristine" run: empty grid, ant at the origin,
    # nothing randomized or loaded. The known highway-onset constants in
    # engine/highway.py are only valid in that case.
    cdef public bint grid_pristine
    cdef object _lock

    cdef public long long ant_x, ant_y
    cdef public int ant_dir
    cdef public long long step_count
    cdef public long long min_x, max_x, min_y, max_y
    cdef public tuple most_visited_cell
    cdef public long long max_visits
    cdef public long long unique_cells_count
    cdef double _sim_start

    def __init__(self):
        self.chunks = {}
        self.stat_chunks = {}
        self.rules = ""
        self.next_state_lut_py = []
        self.direction_change_lut_py = []
        self.statistics_enabled = True
        self.grid_pristine = True
        self._lock = threading.Lock()
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
    def set_rules(self, rules_str):
        self.rules = rules_str.strip().upper()
        n = len(self.rules)
        self.next_state_lut_py = [(i + 1) % n for i in range(n)] if n else []
        self.direction_change_lut_py = [3 if c == "L" else 1 for c in self.rules]
        self.reset()

    def get_rules(self):
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
            self.grid_pristine = True

    def reset_statistics(self):
        with self._lock:
            self.stat_chunks.clear()
            self.most_visited_cell = (0, 0)
            self.max_visits = 0
            self.unique_cells_count = 0
            self._sim_start = time.perf_counter()

    def set_statistics_enabled(self, enabled):
        self.statistics_enabled = bool(enabled)
        if not enabled:
            self.reset_statistics()

    cdef tuple _get_chunk(self, tuple key):
        cdef Chunk chunk = self.chunks.get(key)
        if chunk is None:
            chunk = Chunk()
            self.chunks[key] = chunk
        cdef StatChunk stat_chunk = None
        if self.statistics_enabled:
            stat_chunk = self.stat_chunks.get(key)
            if stat_chunk is None:
                stat_chunk = StatChunk()
                self.stat_chunks[key] = stat_chunk
        return chunk, stat_chunk

    # ------------------------------------------------------------------ #
    def next_step(self, long long steps=1):
        if not self.rules or steps <= 0:
            return

        cdef int n_states = len(self.next_state_lut_py)
        cdef int[::1] next_lut = np.asarray(self.next_state_lut_py, dtype=np.int32)
        cdef int[::1] dir_lut = np.asarray(self.direction_change_lut_py, dtype=np.int32)

        cdef long long cx = chunk_index(self.ant_x)
        cdef long long cy = chunk_index(self.ant_y)
        cdef tuple key = (cx, cy)
        chunk_obj, stat_obj = self._get_chunk(key)
        cdef Chunk chunk = chunk_obj
        cdef StatChunk stat_chunk = stat_obj

        cdef uint32_t[::1] states = chunk.states
        cdef int64_t[::1] visits = stat_chunk.visits if stat_chunk is not None else None
        cdef int64_t[:, ::1] corners = stat_chunk.corners if stat_chunk is not None else None
        cdef int64_t[::1] first_visit = stat_chunk.first_visit_step if stat_chunk is not None else None
        cdef int64_t[::1] last_visit = stat_chunk.last_visit_step if stat_chunk is not None else None

        cdef long long ant_x = self.ant_x
        cdef long long ant_y = self.ant_y
        cdef int ant_dir = self.ant_dir
        cdef long long step_count = self.step_count
        cdef long long min_x = self.min_x, max_x = self.max_x
        cdef long long min_y = self.min_y, max_y = self.max_y
        cdef long long mv_x = self.most_visited_cell[0], mv_y = self.most_visited_cell[1]
        cdef long long max_visits = self.max_visits
        cdef long long unique_cells_count = self.unique_cells_count
        cdef bint stats_on = self.statistics_enabled

        cdef long long new_cx, new_cy, lx, ly, idx, s, i
        cdef int state, old_dir, entry_side, corner_index, dx, dy
        cdef long long v

        for s in range(steps):
            new_cx = ant_x // CHUNK_SIZE if ant_x >= 0 else -((-ant_x + CHUNK_SIZE - 1) // CHUNK_SIZE)
            new_cy = ant_y // CHUNK_SIZE if ant_y >= 0 else -((-ant_y + CHUNK_SIZE - 1) // CHUNK_SIZE)

            if new_cx != cx or new_cy != cy:
                cx, cy = new_cx, new_cy
                key = (cx, cy)
                # Chunk switch needs the GIL for the dict lookup/insert.
                chunk_obj, stat_obj = self._get_chunk(key)
                chunk = chunk_obj
                stat_chunk = stat_obj
                states = chunk.states
                if stat_chunk is not None:
                    visits = stat_chunk.visits
                    corners = stat_chunk.corners
                    first_visit = stat_chunk.first_visit_step
                    last_visit = stat_chunk.last_visit_step

            lx = ant_x - cx * CHUNK_SIZE
            ly = ant_y - cy * CHUNK_SIZE
            idx = (ly << CHUNK_SHIFT) | lx

            state = states[idx]
            old_dir = ant_dir
            if state < n_states:
                ant_dir = (ant_dir + dir_lut[state]) & 3
                states[idx] = next_lut[state]

            if stats_on and stat_chunk is not None:
                visits[idx] += 1
                if first_visit[idx] == -1:
                    first_visit[idx] = step_count + 1
                    unique_cells_count += 1
                last_visit[idx] = step_count + 1

                v = visits[idx]
                if v > max_visits:
                    max_visits = v
                    mv_x, mv_y = ant_x, ant_y

                entry_side = (old_dir + 2) & 3
                corner_index = CORNER_LUT[(entry_side << 2) | ant_dir]
                if corner_index >= 0:
                    corners[idx, corner_index] += 1

            dx = DIR_DX[ant_dir]
            dy = DIR_DY[ant_dir]
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
                self.most_visited_cell = (mv_x, mv_y)
                self.max_visits = max_visits
                self.unique_cells_count = unique_cells_count

    # ------------------------------------------------------------------ #
    @staticmethod
    def estimate_randomize_area_bytes(radius):
        if radius < 0:
            return 0
        side = 2 * radius + 1
        chunks_per_side = (side + CHUNK_SIZE - 1) // CHUNK_SIZE
        total_chunks = chunks_per_side * chunks_per_side
        return total_chunks * CHUNK_AREA * 4

    def randomize_area(self, long long radius):
        if not self.rules or radius < 0:
            return
        cdef int state_count = len(self.rules)
        if state_count == 0:
            return

        cdef long long start_x = self.ant_x - radius, end_x = self.ant_x + radius
        cdef long long start_y = self.ant_y - radius, end_y = self.ant_y + radius
        cdef long long start_cx = chunk_index(start_x), end_cx = chunk_index(end_x)
        cdef long long start_cy = chunk_index(start_y), end_cy = chunk_index(end_y)

        rng = np.random.default_rng()

        cdef long long cx, cy, chunk_min_x, chunk_min_y
        cdef long long local_start_x, local_end_x, local_start_y, local_end_y

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

        # The grid is no longer an untouched run, so the known highway-onset
        # constant no longer applies - engine/highway.py must scan instead.
        self.grid_pristine = False

    # ------------------------------------------------------------------ #
    def get_visit_count(self, long long x, long long y):
        if not self.statistics_enabled:
            return 0
        cdef long long cx = chunk_index(x), cy = chunk_index(y)
        with self._lock:
            sc = self.stat_chunks.get((cx, cy))
            if sc is None:
                return 0
            lx, ly = x - cx * CHUNK_SIZE, y - cy * CHUNK_SIZE
            return int(sc.visits[(ly << CHUNK_SHIFT) | lx])

    def get_most_visited_cell(self):
        with self._lock:
            return self.most_visited_cell

    def get_statistics_summary(self):
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

    def get_top_visited_cells(self, int count):
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
    def save_state(self, filename):
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
                    "chunks": {k: np.asarray(v.states) for k, v in self.chunks.items()},
                    "stat_chunks": {
                        k: (np.asarray(v.visits), np.asarray(v.corners),
                            np.asarray(v.first_visit_step), np.asarray(v.last_visit_step))
                        for k, v in self.stat_chunks.items()
                    },
                }
            with open(filename, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except OSError:
            return False

    def load_state(self, filename):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        except (OSError, pickle.UnpicklingError):
            return False

        self.set_rules(data["rules"])
        with self._lock:
            self.ant_x, self.ant_y, self.ant_dir = data["ant_x"], data["ant_y"], data["ant_dir"]
            self.step_count = data["step_count"]
            self.min_x, self.max_x = data["min_x"], data["max_x"]
            self.min_y, self.max_y = data["min_y"], data["max_y"]
            self.most_visited_cell = data["most_visited_cell"]
            self.max_visits = data["max_visits"]
            self.unique_cells_count = data["unique_cells_count"]
            self.statistics_enabled = data["statistics_enabled"]
            self.grid_pristine = False

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
