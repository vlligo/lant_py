# Langton's Ant — Python port

A Python rewrite of the original Qt/C++ "Langton's Ant with Statistics"
app: a generalized Langton's-Ant / turmite simulator over an unbounded
chunked grid, with per-cell visit/rotation statistics, a pan/zoomable
canvas, and now a TensorBoard dashboard for the statistics.

## Architecture

```
engine/
  common.py          shared constants (chunk geometry, corner LUT, rule
                      string expand/compress helpers) used by both engines
  ant_engine_py.py    pure-Python + NumPy engine — always works, no build
                      step. Also the correctness reference for the Cython one.
  ant_engine_cy.pyx   Cython-accelerated engine — same API, typed/GIL-released
                      inner step loop for ~20-50x speed. Needs a compile step.
  setup.py            builds ant_engine_cy.pyx in place
  loader.py           import AntEngine from here — prefers the compiled
                      engine, falls back to pure Python automatically
  highway.py          "steps until highway": known constants for a pristine
                      grid, plus a lookahead scanner for randomized ones
gui/
  antfieldwidget.py   the pan/zoom/paint canvas (PySide6 QWidget)
  mainwindow.py        the full control panel — 1:1 layout port of the
                       original MainWindow, plus a TensorBoard toggle
  int_spinbox.py       arbitrary-precision-int spin box (Python ints have
                       no overflow limit, unlike Qt's 32-bit QSpinBox)
  highway_worker.py    runs the highway scan on a background thread so a
                       long lookahead never freezes the UI
stats/
  tb_logger.py         wraps torch.utils.tensorboard.SummaryWriter;
                       no-ops cleanly if torch isn't installed
tests/
  test_engine.py       correctness tests + a Python-vs-Cython parity test
main.py                 entry point
```

### Why Cython, and how the speed actually comes from it

The ant's next state depends on the previous step, so the simulation is
inherently **sequential** — you cannot vectorize "steps" across time the
way you would a batch of independent computations. The only lever is
making each individual step cheap. Cython gets you there by:

- Fully typing the ant's position/direction/step-count as C integers
  (`long long`, `int`) instead of Python objects.
- Using typed memoryviews (`uint32_t[::1]`, `int64_t[::1]`) over each
  chunk's NumPy arrays, so cell reads/writes inside the loop compile to
  raw array indexing, not Python `__getitem__`.
- Caching the current chunk's memoryviews across steps and only touching
  the (GIL-requiring) Python dict when the ant actually crosses a chunk
  boundary — which happens roughly once every 32 steps, not every step.

This mirrors the original C++'s own optimization (it caches
`currentChunk`/`currentStatChunk` pointers for exactly the same reason).

### Building the accelerated engine

```bash
pip install -r requirements.txt
cd engine
python setup.py build_ext --inplace
cd ..
python main.py
```

> **If you already built the extension before the highway feature was
> added, rebuild it.** `ant_engine_cy.pyx` gained a `grid_pristine` flag,
> and a stale `.so` won't have it.

If you skip the build step, the app runs on the pure-Python engine
automatically (you'll see a one-time warning in the log) — correct, just
much slower for large step counts. `engine/loader.py` is the only place
that decides which one gets imported; nothing else in the app needs to
change either way.

### Verifying the port

```bash
python tests/test_engine.py
```

Runs invariant checks against the pure-Python engine, and — if you've
built the Cython extension — a parity test that runs both engines on
several rule strings and asserts they land on identical ant position,
step count, unique-cell count, max-visit count, and top-20 cells.

## TensorBoard statistics

Toggle it from the **Stats Logging** menu in the app, or drive it
headlessly:

```python
from engine.loader import AntEngine
from stats.tb_logger import TensorBoardLogger

engine = AntEngine()
engine.set_rules("LR")
logger = TensorBoardLogger(run_name="LR_run")

for _ in range(200):
    engine.next_step(50_000)
    logger.log_summary(engine.get_statistics_summary(), engine.get_rules())

logger.close()
```

Then:

```bash
tensorboard --logdir runs
```

Each run gets its own timestamped subfolder under `runs/`, so you can
compare different rule strings (or Python vs. Cython run speed) side by
side on the same dashboard. Logged scalars: `unique_cells_visited`,
`max_visits_per_cell`, `average_visits`, `total_steps`, and a derived
`steps_per_second`; the rule string is logged as text so you can tell
runs apart in the UI.

## Steps until highway

The classic LR ant wanders chaotically and then abruptly locks into a
**highway**: a 104-step cycle that repeats forever, translating the ant 2
cells diagonally each period. The Quick Statistics panel shows a live
countdown to it, plus a **Rescan Highway** button.

There are two genuinely different situations, and the app handles them
differently because only one of them has an answer you can look up:

**Pristine grid** (empty, ant at origin, nothing randomized or loaded).
The rule is deterministic and the start state is fixed, so the onset is a
*constant*. It is in `KNOWN_HIGHWAYS` in `engine/highway.py`:

| rule | onset step | period | translation |
|------|-----------|--------|-------------|
| LR   | 9,977     | 104    | (+2, +2)    |
| RL   | 9,977     | 104    | (-2, +2)    |

Both were measured with `detect_highway_onset()` in that module, not
taken from memory, and there is a test asserting the independent
lookahead scanner rediscovers the same numbers.

**Randomized or loaded grid.** Here there is no constant to look up —
the onset depends on every single random cell. Three different random
areas in testing produced onsets of 903, 18,326 and 85,322 steps, with
three different travel directions. So the only way to answer "how many
steps left" is to *actually simulate ahead*, which is what
`scan_for_highway()` does:

1. Clones the engine into a throwaway shadow copy (statistics dropped —
   a StatChunk is ~56 KB against a Chunk's 4 KB, so this makes the clone
   ~14x smaller and the scan faster). The live simulation is untouched.
2. Runs the shadow forward **one period at a time**, sampling the ant's
   position and direction at each period boundary. Sampling per period
   rather than per step is what keeps it cheap: the 104 steps between
   samples run inside compiled code, and Python only inspects ~1% as
   often as there are steps.
3. Accepts the highway once 8 consecutive periods agree on both direction
   and a constant non-zero translation, then walks back to the earliest
   consistent period and single-steps a short window to pin the *exact*
   onset step.
4. Keeps going until the ant has provably escaped every pre-existing cell
   in its direction of travel (see `certain`, below).

The scan runs on a background thread (`gui/highway_worker.py`), so the UI
never freezes; a new scan cancels any in-flight one, so a stale result
can't overwrite a fresh configuration. It is triggered on rule change,
reset, randomize, and load — **not** on every step, because the onset is
an absolute step number, so the live countdown is pure subtraction.

### Two caveats the UI is honest about

- **"Provisional" results.** A pattern can look periodic and still be
  destroyed later if the ant ploughs back into cells that were already
  set — your randomized area, or its own earlier chaotic blob. The
  scanner therefore continues past detection until the ant's footprint
  has cleared the bounding box of all pre-existing cells *in its
  direction of travel*, which makes the highway permanent by induction
  (every future period lands on virgin cells or its own translated
  structure). Until that holds, the label reads "(provisional)".

- **"Not found" is a legitimate answer.** That the ant *always*
  eventually builds a highway is an open conjecture, not a theorem. The
  scan runs under a step budget and will say so rather than invent a
  number. Likewise, if you scan while already inside a highway, a
  forward-only scan cannot recover when it started, so the label says
  "already active" instead of claiming a false onset.

### Scan budget

The budget is selectable in the Quick Statistics panel — Quick (1M),
Normal (100M, the default), Long (1B), Exhaustive (10B), or a custom
value. While a scan runs, the label shows live progress and the button
becomes **Cancel Scan**.

Two things worth knowing:

- **A large budget is usually free.** The scan stops the instant it
  detects a highway, so the budget only bites when there *isn't* one to
  find. Raising it never changes an answer, it only reveals one — there's
  a test asserting exactly that.
- **Budget is not the same as time.** The pure-Python engine manages
  ~0.5M steps/sec; the compiled Cython engine is roughly 20-50x that. So
  a 100M-step budget is a few seconds with Cython and a few minutes
  without. That asymmetry is why the cancel button exists — if you're
  running the fallback engine, prefer Quick or Normal.

Memory is flat in the budget. It didn't used to be: the scan originally
retained every period checkpoint, costing ~2 MB per million steps (~1 GB
at a 500M budget), which is what made large budgets impractical. Since
the periodicity window is tested after *every* period, the first window
that comes back consistent is already the earliest one, so nothing before
the current streak is ever needed — only the running streak is carried
forward. An 8M-step scan now peaks at ~0.12 MB, and
`test_scan_memory_does_not_grow_with_budget` guards against regressing
it. The onset the O(1) version reports was checked against an exhaustive
per-step ground truth and matches exactly, on both pristine and
randomized grids.

The escape-confirmation budget (how long the scanner keeps going to prove
the ant cleared all pre-existing cells) scales with the main budget
rather than being pinned to a constant, since a large randomized area
takes proportionally longer to escape.

Other rule strings are not wired up because the period is rule-specific
(many rules never form a highway at all). The detector itself is generic:
pass a different `period` to `scan_for_highway()` to extend it.

## What's simplified vs. the original

- **Save/load format**: uses Python's `pickle` instead of the original's
  raw `QDataStream` binary layout. Same information, not byte-compatible
  with old `.ant` files from the C++ app.
- **`Rotations`/`Arcs`/`Diagonals` display styles**: ported faithfully
  (same geometry/parity logic as the original `redrawBuffer()`), but they
  scan/draw every visited cell in the visible chunks each repaint exactly
  like the original did — expect them to get slow at very high zoom-in
  over heavily-visited areas, same as before.
- **Multithreaded fills**: the original used `QtConcurrent` to fan a
  `randomizeArea()` fill and the zoomed-out pixel render across threads.
  The Python port gets its speed from NumPy vectorization instead (fewer,
  bigger array ops rather than many small parallel tasks) — simpler code,
  comparable wall-clock time for the sizes this app is used at. If you
  push `randomizeArea` radius into the hundreds of thousands, that's the
  one place worth profiling first.
