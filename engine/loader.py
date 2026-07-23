"""
Import this module's `AntEngine` and `ENGINE_BACKEND` everywhere else in
the app (GUI, stats logger, CLI). It transparently prefers the compiled
Cython engine (fast) and falls back to the pure-Python one (always works,
no build step) if the extension hasn't been built yet.
"""
import logging

log = logging.getLogger(__name__)

try:
    from .ant_engine_cy import AntEngine, ENGINE_BACKEND  # type: ignore
except ImportError:
    from .ant_engine_py import AntEngine, ENGINE_BACKEND
    log.warning(
        "Compiled Cython engine not found — using the pure-Python fallback "
        "(much slower). Build it with: cd engine && python setup.py build_ext --inplace"
    )

__all__ = ["AntEngine", "ENGINE_BACKEND"]
