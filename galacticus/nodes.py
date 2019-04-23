#! /usr/bin/env python

import sys,os,fnmatch,re,copy
import numpy as np
import warnings
from . import rcParams
from .datasets import Dataset
from .properties.manager import Property


def getHostIndex(nodeIsIsolated):
    nodes = nodeIsIsolated[::-1]
    def getIndex(values,i):
        global previous
        if values[i] == 1:
            previous = len(values)-i-1
        return previous
    result = np.array([getIndex(nodes,i)for i in range(len(nodes))])
    result = result[::-1]
    return result


@Property.register_subclass('hostNode')
class HostNode(Property):
    
    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        return

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        Function to identify whether this class can process a specified property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if propertyName.endswith(":host"):
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+"' is not valid."+\
                "Host node properties must end with ':host'."
            raise RuntimeError(msg)
        return False
    
    def get(self,propertyName,redshift):
        """                                                                                                                                                                                                                                        
        Return property of a host node.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Get name of original property
        nodeProperty = propertyName.replace(":host","")
        # Read appropriate properties
        GALS = self.galaxies.get(redshift,properties=[nodeProperty,"nodeIsIsolated"])
        # Check for 'empty' node property
        if GALS[nodeProperty] is None:
            return None
        # Locate indices of hosts
        hostIndex = getHostIndex(GALS["nodeIsIsolated"].data)
        # Construct Dataset object
        DATA = Dataset(name=propertyName)
        DATA.attr = copy.copy(GALS[nodeProperty].attr)
        DATA.data = np.copy(GALS[nodeProperty].data)[hostIndex]
        return DATA
