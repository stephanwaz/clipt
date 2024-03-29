# -*- coding: utf-8 -*-

"""Top-level package for clipt."""

__author__ = """Stephen Wasilewski"""
__email__ = 'stephanwaz@gmail.com'
__version__ = '1.0.15'
__all__ = ['plot', 'cl_plot']


import matplotlib
backend = matplotlib.get_backend()
import clipt.plot as mplt

try:
    matplotlib.use(backend)
except ModuleNotFoundError:
    pass


def get_root(inside=True, pyenv=False):
    """return root directory of clipt install"""

    import sys

    if pyenv:
        return sys.executable.rsplit("/bin", 1)[0]
    elif inside:
        return __file__.rsplit("clipt", 1)[0] + "clipt"
    else:
        return __file__.rsplit("/clipt", 1)[0]