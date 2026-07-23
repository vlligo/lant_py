"""
TensorBoard logging for the ant simulation.

Uses torch.utils.tensorboard.SummaryWriter purely as a writer — no model
training involved, just a convenient, well-known dashboard for scalars
and histograms. View with:

    tensorboard --logdir runs

Then open the printed localhost URL. Scalars land under "ant/", so you
can compare multiple rule strings/runs side by side in one dashboard
(each run gets its own subfolder + timestamp).
"""
import time
from pathlib import Path
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


class TensorBoardLogger:
    """Wraps a SummaryWriter and throttles writes to avoid slowing the sim.

    Safe to construct even if torch/tensorboard aren't installed — logging
    calls silently become no-ops (`enabled` stays False) so the rest of the
    app doesn't need to special-case it.
    """

    def __init__(self, log_dir: str = "runs", run_name: Optional[str] = None,
                 min_interval_sec: float = 0.5):
        self.enabled = False
        self.writer = None
        self._last_write = 0.0
        self.min_interval_sec = min_interval_sec

        if not TENSORBOARD_AVAILABLE:
            return

        run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        path = Path(log_dir) / run_name
        path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(path))
        self.enabled = True
        self.run_path = str(path)

    def log_summary(self, summary, rules: str, force: bool = False) -> bool:
        """Log the current AntStatisticsSummary as TensorBoard scalars.

        Throttled to `min_interval_sec` unless force=True. Returns whether
        a write actually happened (useful for UI feedback / tests).
        """
        if not self.enabled:
            return False

        now = time.monotonic()
        if not force and (now - self._last_write) < self.min_interval_sec:
            return False
        self._last_write = now

        step = summary.total_cells_visited
        self.writer.add_scalar("ant/unique_cells_visited", summary.unique_cells_visited, step)
        self.writer.add_scalar("ant/max_visits_per_cell", summary.max_visits_per_cell, step)
        self.writer.add_scalar("ant/average_visits", summary.average_visits, step)
        self.writer.add_scalar("ant/total_steps", summary.total_cells_visited, step)
        if summary.simulation_time_ms > 0:
            steps_per_sec = summary.total_cells_visited / (summary.simulation_time_ms / 1000.0)
            self.writer.add_scalar("ant/steps_per_second", steps_per_sec, step)
        self.writer.add_text("ant/rules", rules, step)
        return True

    def log_visit_histogram(self, engine, step: int, max_samples: int = 200_000):
        """Log a histogram of per-cell visit counts. Call this occasionally
        (not every frame) — it walks every tracked chunk."""
        if not self.enabled:
            return
        import numpy as np
        counts = []
        for sc in engine.stat_chunks.values():
            nz = np.asarray(sc.visits)
            nz = nz[nz > 0]
            counts.append(nz)
            if sum(len(c) for c in counts) > max_samples:
                break
        if not counts:
            return
        all_counts = np.concatenate(counts)
        self.writer.add_histogram("ant/visit_distribution", all_counts, step)

    def flush(self):
        if self.enabled:
            self.writer.flush()

    def close(self):
        if self.enabled:
            self.writer.close()
            self.enabled = False
