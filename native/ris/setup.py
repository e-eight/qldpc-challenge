"""Build the standalone extension; RIS_NATIVE=1 opts into host CPU instructions."""

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="ris-native",
    version="0.1.0",
    python_requires=">=3.10",
    install_requires=["numpy>=2"],
    packages=[],
    ext_modules=[
        Pybind11Extension(
            "ris_native",
            ["src/bindings.cpp", "src/ris.cpp"],
            include_dirs=["include"],
            cxx_std=17,
            define_macros=[("RIS_HOST_TUNED", "1" if os.getenv("RIS_NATIVE") == "1" else "0")],
            extra_compile_args=[
                "-O3",
                "-g",
                "-pthread",
                *(["-march=native"] if os.getenv("RIS_NATIVE") == "1" else []),
            ],
            extra_link_args=["-pthread"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
