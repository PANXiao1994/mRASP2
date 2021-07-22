from distutils.core import setup
from Cython.Build import cythonize
import numpy

extra_compile_args = ["-stdlib=libc++", "-O3"]


setup(
    ext_modules = cythonize("mcolt/data/data_utils_fast.pyx"),
    extra_compile_args=extra_compile_args,
    include_dirs=[numpy.get_include()]
)
