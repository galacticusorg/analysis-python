#! /usr/bin/env python

import six
import copy
import numpy as np
from . import GalacticusParameters
from ..fileFormats.hdf5 import HDF5
from ..strings import removeByteStrings,addByteStrings

class ParametersFromHDF5(object):
    
    @classmethod
    def addGroupParameters(cls,GH5,OUT,PARAMS):
        path = copy.copy(OUT.name)
        path = path.replace("/Parameters","/parameters")
        if len(OUT.attrs.keys())>0:
            for key in OUT.attrs.keys():
                value = copy.copy(OUT.attrs[key])
                if six.PY3:
                    value = removeByteStrings(value)
                PARAMS.setParameter(path+"/"+key,value,createParents=True)
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
        

class ParametersToHDF5(object):
    
    @classmethod
    def writeParameter(cls,GH5,path,param,append=True,overwrite=False):
        hdfdir = path.replace("/parameters","/Parameters")
        if param is None:                        
            GH5.mkGroup(hdfdir)
        else:
            name = hdfdir.split("/")[-1]
            hdfdir = hdfdir.replace("/"+name,"")
            GH5.mkGroup(hdfdir)
            attr = GH5.readAttributes(hdfdir)
            if name in attr.keys():
                if append:
                    cls.append(GH5,path,param)
                if overwrite:
                    GH5.rmAttributes(hdfdir,attributes=[name])
                    attr = GH5.readAttributes(hdfdir)
            if name not in attr.keys():
                GH5.addAttributes(hdfdir,{name:param})
        return

    @classmethod
    def append(cls,GH5,path,param):
        hdfdir = path.replace("/parameters","/Parameters")
        name = hdfdir.split("/")[-1]
        hdfdir = hdfdir.replace("/"+name,"")
        values = GH5.readAttributes(hdfdir,required=[name])[name]        
        if np.ndim(existing) == 0:
            values = [values]
        values.append(param)
        GH5.addAtrributes(hdfdir,{name:values})
        return
        
    @classmethod
    def write(cls,GH5,PARAMS,append=True,overwrite=False):
        if "Parameters" not in GH5.lsGroups("/"):
            GH5.mkGroup("/Parameters")
            paths = list(set(PARAMS.map).difference(["/","/parameters"]))
            [cls.writeParameter(GH5,path,PARAMS.getParameter(path),append=append,
                                overwrite=overwrite) for path in paths]
        return



