from setuptools import Extension, setup
import numpy as np

extensions = [Extension('hh_cython',
                        ['hh_cython.pyx'],
                        include_dirs = [np.get_include()])]
from Cython.Build import cythonize
setup(
    ext_modules = cythonize(extensions)
)