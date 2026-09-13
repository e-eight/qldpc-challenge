from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="ris-guided-prototype", version="0.0.1", packages=[],
    ext_modules=[Pybind11Extension(
        "ris_guided_native", ["src/bindings.cpp", "src/ris.cpp"],
        include_dirs=["include"], cxx_std=17,
        define_macros=[("RIS_HOST_TUNED", "1")],
        extra_compile_args=["-O3", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    )], cmdclass={"build_ext": build_ext},
)
