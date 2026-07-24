"""
Run with: python -m pytest tests/  (or plain `python tests/test_engine.py`)

test_cython_parity() only runs if the Cython extension has been built
(engine/setup.py) — it's skipped otherwise, since the pure-Python engine
is always available but the compiled one is opt-in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.ant_engine_py import AntEngine as PyEngine


def _basic_invariants(engine, rules="RL", steps=5000):
    engine.set_rules(rules)
    engine.next_step(steps)
    assert engine.step_count == steps
    # Every step visits exactly one cell, so total visits == step_count.
    assert engine.unique_cells_count <= steps
    summary = engine.get_statistics_summary()
    assert summary.total_cells_visited == steps
    if summary.unique_cells_visited:
        assert abs(summary.average_visits - steps / summary.unique_cells_visited) < 1e-9
    top = engine.get_top_visited_cells(5)
    assert all(top[i][1] >= top[i + 1][1] for i in range(len(top) - 1)), "top cells not sorted desc"
    return engine


def test_python_engine_classic_rule():
    _basic_invariants(PyEngine(), "RL", 11000)


def test_python_engine_multi_state_rule():
    _basic_invariants(PyEngine(), "LLRRRLRLRLLR", 5000)


def test_save_load_roundtrip(tmp_path):
    e = PyEngine()
    e.set_rules("LR")
    e.next_step(777)
    e.randomize_area(15)
    path = str(tmp_path / "state.ant")
    assert e.save_state(path)

    e2 = PyEngine()
    assert e2.load_state(path)
    assert e2.step_count == e.step_count
    assert (e2.ant_x, e2.ant_y, e2.ant_dir) == (e.ant_x, e.ant_y, e.ant_dir)
    assert e2.get_top_visited_cells(10) == e.get_top_visited_cells(10)


def test_cython_parity():
    try:
        from engine.ant_engine_cy import AntEngine as CyEngine
    except ImportError:
        try:
            import pytest
            pytest.skip("Cython extension not built — run: cd engine && python setup.py build_ext --inplace")
        except ImportError:
            print("Cython extension not built — skipping parity test.")
        return

    for rules in ("RL", "LLRR", "LLRRRLRLRLLR", "LRRRRRLLR"):
        py_e, cy_e = PyEngine(), CyEngine()
        py_e.set_rules(rules)
        cy_e.set_rules(rules)
        py_e.next_step(20000)
        cy_e.next_step(20000)

        assert (py_e.ant_x, py_e.ant_y, py_e.ant_dir) == (cy_e.ant_x, cy_e.ant_y, cy_e.ant_dir)
        assert py_e.unique_cells_count == cy_e.unique_cells_count
        assert py_e.max_visits == cy_e.max_visits
        assert py_e.get_top_visited_cells(20) == cy_e.get_top_visited_cells(20)


if __name__ == "__main__":
    test_python_engine_classic_rule()
    test_python_engine_multi_state_rule()
    print("All pure-Python engine tests passed.")
    test_cython_parity()


# --------------------------------------------------------------------- #
# Highway detection
# --------------------------------------------------------------------- #
def test_known_constant_matches_scanner_on_pristine_grid():
    """The hardcoded constant and the lookahead scanner must agree."""
    from engine.highway import KNOWN_HIGHWAYS, scan_for_highway

    e = PyEngine()
    e.set_rules("LR")
    status = scan_for_highway(e, max_steps=100_000)
    onset, period, translation = KNOWN_HIGHWAYS["LR"]

    assert status.known and status.onset_step == onset
    assert status.period == period and status.displacement == translation
    assert status.certain


def test_pristine_flag_cleared_by_randomize():
    e = PyEngine()
    e.set_rules("LR")
    assert e.grid_pristine
    e.randomize_area(10)
    assert not e.grid_pristine, "randomizing must invalidate the highway constant"
    e.reset()
    assert e.grid_pristine


def test_countdown_is_pure_subtraction():
    from engine.highway import scan_for_highway

    e = PyEngine()
    e.set_rules("LR")
    status = scan_for_highway(e, max_steps=100_000)
    assert status.for_step(0).steps_remaining == status.onset_step
    assert status.for_step(status.onset_step - 100).steps_remaining == 100
    reached = status.for_step(status.onset_step + 500)
    assert reached.reached and reached.steps_since_onset == 500


def test_scan_finds_highway_after_randomize():
    """The real point of the scanner: no constant exists here, and the
    predicted onset must genuinely be the start of a periodic highway."""
    import numpy as np
    from engine.highway import scan_for_highway

    np.random.seed(7)
    e = PyEngine()
    e.set_rules("LR")
    e.randomize_area(20)

    status = scan_for_highway(e, max_steps=400_000)
    assert status.known, "no highway found on the randomized grid"
    assert status.onset_step is not None

    # Independently verify by running the LIVE engine to the predicted
    # onset and checking the trajectory really is periodic from there.
    e.next_step(status.onset_step)
    period = status.period
    dx, dy = status.displacement
    traj = []
    for _ in range(period * 4):
        traj.append((e.ant_x, e.ant_y, e.ant_dir))
        e.next_step(1)
    for i in range(period * 3):
        xa, ya, da = traj[i]
        xb, yb, db = traj[i + period]
        assert da == db and (xb - xa, yb - ya) == (dx, dy), \
            f"trajectory not periodic at offset {i}"


def test_larger_budget_finds_what_a_small_one_misses():
    """The budget is the only thing standing between 'not found' and a
    result — raising it must never change the answer, only reveal it."""
    from engine.highway import scan_for_highway

    e = PyEngine()
    e.set_rules("LR")

    tiny = scan_for_highway(e, max_steps=2_000)
    assert not tiny.known, "2,000 steps should not reach the ~9,977 onset"
    assert tiny.budget_exhausted or "No highway found" in tiny.reason

    ample = scan_for_highway(e, max_steps=200_000)
    assert ample.known and ample.onset_step == 9977

    # An even bigger budget must agree, not drift.
    bigger = scan_for_highway(e, max_steps=1_000_000)
    assert bigger.onset_step == ample.onset_step
    assert bigger.displacement == ample.displacement


def test_scan_memory_does_not_grow_with_budget():
    """Regression guard: checkpoint bookkeeping used to be retained for the
    whole scan (~2 MB per million steps), which made large budgets
    unusable. It must now be O(1) in the budget."""
    import tracemalloc
    from engine.highway import scan_for_highway

    def peak_for(budget):
        e = PyEngine()
        e.set_rules("LLRR")          # never forms a period-104 highway
        tracemalloc.start()
        scan_for_highway(e, max_steps=budget)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    small = peak_for(500_000)
    large = peak_for(4_000_000)      # 8x the steps
    assert large < small * 3, (
        f"scan memory scales with budget: {small} -> {large} bytes")


def test_escape_budget_scales_with_main_budget():
    """A huge randomized area needs proportionally longer to confirm escape,
    so the escape budget must not be pinned to a constant."""
    import inspect
    from engine.highway import scan_for_highway

    default = inspect.signature(scan_for_highway).parameters["escape_budget_steps"].default
    assert default is None, "escape budget should be derived from max_steps"
