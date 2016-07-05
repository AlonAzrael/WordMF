import pyximport
from pyximport import install
import numpy as np
PYXIMPORTCPP_FLAG = True

if PYXIMPORTCPP_FLAG:
    old_get_distutils_extension = pyximport.pyximport.get_distutils_extension
    def new_get_distutils_extension(modname, pyxfilename, language_level=None):
        extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
        extension_mod.language='c++'
        extension_mod.extra_compile_args=["-std=c++11"]
        return extension_mod,setup_args
    pyximport.pyximport.get_distutils_extension = new_get_distutils_extension
    install(setup_args={"include_dirs":np.get_include(), })
else:
    pyximport.install(setup_args={"include_dirs":np.get_include(), })


from word_cooc_counter import *
