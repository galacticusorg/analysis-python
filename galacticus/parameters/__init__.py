#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
from ..fileFormats.xmlTree import xmlTree


class GalacticusParameters(xmlTree):
    
    """
    GalacticusParameters(): class to store parameters from a Galacticus parameters file.
    
        Base class: xmlTree
    
        Functions: 
                    getParameter(): Return value of specified parameter.
                    getParameterPath(): Return path to specified parameter in XML tree
                    setParameter(): Set the value for a specified parameter.
                    removeParameter(): Remove specified parameter from the XML tree.

    """    
    def __init__(self,file=None,root='parameters',verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(GalacticusParameters,self).__init__(file=file,root=root)
        return

    def getParameterPath(self,name):
        """
        GalacticusParameters.getParameterPath: Return the path to the parameter in the 
                                               XML tree.
        
        USAGE: value = GalacticusParameters.getParameter(path)

            INPUT
                 path -- Path to parameter, including parameter name
            OUTPUT
                value -- String with value for parameter

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        matches = self.matchPath("/*"+name)
        if len(matches) == 0:
            raise ValueError(funcname+"(): Parameter '"+name+"' cannot be located!")
        return matches[0]
    
    def getParameter(self,path):
        """
        GalacticusParameters.getParameter: Return value of specified parameter.
        
        USAGE: value = GalacticusParameters.getParameter(path)

            INPUT
                 path -- Path to parameter, including parameter name
            OUTPUT
                value -- String with value for parameter

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        value = self.getElementAttribute(path,attrib="value")
        if value is None:            
            print("WARNING! "+funcname+"(): Parameter at path '"+\
                      path+"' cannot be located!")
        return value
    
    def setParameter(self,path,value,createParents=False): 
        """
        GalacticusParameters.setParameter(): Update parameter value.

        USAGE: GalacticusParameters.setParameter(param,value,[createParents=False])
        
           INPUT
               path          -- Path to parameter, including parameter name
               value         -- Value to assign to parameter.
               createParents -- Create the parents in the parameter tree.

        """    
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Convert paramter value to string
        if np.ndim(value) == 0:
            value = str(value)
        else:
            value = " ".join(map(str,value))
        # Set parameter
        self.updateElement(path,attrib={"value":value},createParents=createParents)
        return

    
    def removeParameter(self,path):
        """
        GalacticusParameters.removeParameter(): Remove a parameter from a file.
        
        USAGE:  GalacticusParameters.removeParameter(param)

            INPUT
               path -- Path to parameter, including parameter name

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.removeElement(path)
        return
        
