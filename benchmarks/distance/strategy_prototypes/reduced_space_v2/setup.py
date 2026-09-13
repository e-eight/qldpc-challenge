from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="reduced-space-prototype",
    version="0.0.1",
    packages=[],
    ext_modules=[
        Pybind11Extension(
            "reduced_space_native",
            ["search.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3", "-march=native"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
