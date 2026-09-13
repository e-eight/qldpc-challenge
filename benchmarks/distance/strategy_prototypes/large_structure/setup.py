from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="large-structure-probe",
    version="0.0.1",
    packages=[],
    ext_modules=[
        Pybind11Extension(
            "large_structure_native", ["search.cpp"], cxx_std=17, extra_compile_args=["-O3", "-march=native"]
        )
    ],
    cmdclass={"build_ext": build_ext},
)
