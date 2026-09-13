from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="strategy-structure",
    ext_modules=[Pybind11Extension("_strategy_structure", ["core.cpp"], cxx_std=17,
                                  extra_compile_args=["-O3", "-march=native"])],
    cmdclass={"build_ext": build_ext},
)
