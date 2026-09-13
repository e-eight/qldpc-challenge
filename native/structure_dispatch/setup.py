from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="structure-dispatch",
    version="0.1.0",
    packages=[],
    ext_modules=[
        Pybind11Extension(
            "structure_dispatch_native", ["detect.cpp"], cxx_std=17, extra_compile_args=["-O3", "-march=native"]
        )
    ],
    cmdclass={"build_ext": build_ext},
)
