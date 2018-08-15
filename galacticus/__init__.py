#! /usr/bin/env python

import __future__
import warnings
import os,sys
from configparser import ConfigParser
import pkg_resources

class rcConfig(ConfigParser):
    """
    rcConfig: Class to load and manage a configuration file of rc keyword parameters
              for calculations within the Galacticus python package.

    Base class: configparser.ConfigParser
    
    Functions:
            update(): Updates the specified parameter.

    """
    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(rcConfig,self).__init__()
        configKeyword = "GALACTICUS_PYTHON_CONFIG"
        if configKeyword in os.environ.keys():
            configFile = os.environ[configKeyword]
            warnings.warn("WARNING: Loading non-standard rcPrams configuration."\
                              " [Configuration file: "+configFile+"]")
        else:
            configFile = pkg_resources.resource_filename(__name__,"rcParams.cfg")
        # Read selected configuration file
        self.read(configFile)
        # Store the path to the configuration file
        self.file = configFile
        return

    def __call__(self):
        for path in ["GALACTICUS_EXEC_PATH","GALACTICUS_DATA_PATH"]:
            if path in os.environ.keys():
                self.update("paths",path,os.environ[path])
        return


    def update(self,section,parameter,value):
        """
        rcConfig.update: Updates the parameter in the specified section of the configuration file
                         by assigning the parameter the specified value.

        USAGE: rcConfig.update(section,parameter,value)

          INPUTS
              section   -- The name of the section in the configuration file in which the parameter 
                           can be found. 
              parameter -- The name of the keyword parameter to update.
              value     -- The value(s) to assign to the keyword parameter (accepts scalar or 
                           vector input).

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.set(section,parameter,str(value))
        return

    def reset(self):
        """
        rcConfig.reset: Restore default parameters by re-reading the configuration file.

        USAGE: rcConfig.reset()

        """
        self.read(self.file)
        self.__call__()
        return
        

rcParams = rcConfig()
rcParams()
    

