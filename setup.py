#!/usr/bin/env python
#coding=utf-8

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy 

ext_modules = [Extension("warps", ["warps.pyx"])]

setup(
  name = 'Hello world app',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [numpy.get_include()],
  ext_modules = ext_modules
)
