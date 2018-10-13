#! /usr/bin/env python

import six
import copy
import numpy as np
from . import GalacticusParameters
from ..fileFormats.hdf5 import HDF5
from ..strings import removeByteStrings

class ParametersFromHDF5(object):
    
    @classmethod
    def addGroupParameters(cls,GH5,OUT,PARAMS):
        path = copy.copy(OUT.name)
        path = path.replace("Parameters","parameters")
        if len(OUT.attrs.keys())>0:
            for key in OUT.attrs.keys():
                value = copy.copy(OUT.attrs[key])
                if six.PY3:
                    value = removeByteStrings(value)
                PARAMS.setParameter(path+"/"+key,OUT.attrs[key],createParents=True)
        grps = GH5.lsGroups(OUT.name)
        if len(grps) > 0:
            [cls.addGroupParameters(GH5,GH5.fileObj[OUT.name+"/"+grp],PARAMS) 
             for grp in grps]
        return
    
    @classmethod
    def read(cls,GH5):
        PARAMS = GalacticusParameters()
        PARAMS.mapTree()
        cls.addGroupParameters(GH5,GH5.fileObj["/Parameters"],PARAMS)
        return PARAMS
        

            



