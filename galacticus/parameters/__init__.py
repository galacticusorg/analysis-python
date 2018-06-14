#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from ..fileFormats.xmlTree import xmlTree


class GalacticusParameters(xmlTree):
    
    """
    GalacticusParameters(): class to store parameters from a Galacticus parameters file.
    
        Base class: xmlTree
    
        Functions: 
                    getParameter(): Return value of specified parameter.
                    getParent(): Return name of the parent XML element for specified parameter.
                    constructDictionary(): Build dictionary of parameters.
                    setParameter(): Set the value for a specified parameter.

    """
    
    def __init__(self,xmlfile,root='parameters',verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(GalacticusParameters,self).__init__(xmlfile=xmlfile,root=root,verbose=verbose)
        return
    
    def getParameter(self,param):
        """
        get_parameter: Return value of specified parameter.
        
        USAGE: value = getParameter(param)

            INPUT
                param -- name of parameter
            OUTPUT
                value -- string with value for parameter

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if not param in self.treeMap.keys():
            if self._verbose:
                print("WARNING! "+funcname+"(): Parameter '"+\
                          param+"' cannot be located in "+self.xmlfile)
            value = None
        else:
            path = self.treeMap[param]
            elem = self.getElement(path)
            if elem is not None:
                value = elem.attrib.get("value")
            else:
                value = None
        return value

    def getParent(self,param):
        """
        getParent: Return name of parent element.

        USAGE: name = getParent(elem)

            INPUT
                elem -- name of current element
            OUTPUT
                name -- string with name of parent

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not param in self.treeMap.keys():
            if self._verbose:
                print("WARNING! "+funcname+"(): Parameter '"+\
                          param+"' cannot be located in "+self.xmlfile)
            name = None
        else:
            name = self.treeMap[param]
            name = name.split("/")[-2]
        return name

    def constructDictionary(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        params = {}
        for e in self.treeMap.keys():
            params[e] = self.getParameter(e)
        return params

    
    def setParameter(self,param,value,parent=None,selfCreate=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Convert paramter value to string
        if np.ndim(value) == 0:
            value = str(value)
        else:
            value = " ".join(map(str,value))
        # Set parameter
        self.setElement(param,attrib={"value":value},parent=parent,\
                            selfCreate=selfCreate)
        return

    
    def removeParameter(self,param):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.removeElement(param)
        return
        

def formatParametersFile(ifile,ofile=None):    
    import shutil
    tmpfile = ifile.replace(".xml","_copy.xml")
    if ofile is not None:
        cmd = "xmllint --format "+ifile+" > "+ofile
    else:
        cmd = "xmllint --format "+ifile+" > "+tmpfile
    os.system(cmd)
    if ofile is None:
        shutil.move(tmpfile,ifile)        
    return

