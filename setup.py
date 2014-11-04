from distutils.core import setup
from distutils.core import Extension
#from Cython.Build import cythonize
from utils.cyutil import cythonize
import numpy

setup(
    ext_modules=cythonize("**/*.pyx"),
    include_dirs=[numpy.get_include()]
    )
