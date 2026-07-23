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
gui/
  antfieldwidget.py   the pan/zoom/paint canvas (PySide6 QWidget)
  mainwindow.py        the full control panel — 1:1 layout port of the
                       original MainWindow, plus a TensorBoard toggle
  int_spinbox.py       arbitrary-precision-int spin box (Python ints have
                       no overflow limit, unlike Qt's 32-bit QSpinBox)
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
