"""
Build the Cython-accelerated engine in place:

    cd engine
    python setup.py build_ext --inplace

This produces ant_engine_cy*.so (Linux/Mac) or .pyd (Windows) next to
this file. The app will pick it up automatically the next time it runs
(see engine/loader.py) — no other changes needed.

Requires: cython, numpy, and a C compiler (gcc/clang on Linux/Mac,
MSVC Build Tools on Windows).
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "ant_engine_cy",
        ["ant_engine_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="ant_engine_cy",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    zip_safe=False,
)
