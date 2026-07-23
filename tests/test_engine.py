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
