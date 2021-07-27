import os
import numpy

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except:
        raise OSError("Can't load library %s" % libname)

