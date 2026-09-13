"""Build the isolated stabilizer-descent experiment."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="strategy-stabilizer-descent",
    version="0.0.1",
    packages=[],
    ext_modules=[Pybind11Extension(
        "_stabilizer_descent", ["descent.cpp"], cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    )],
    cmdclass={"build_ext": build_ext},
)
