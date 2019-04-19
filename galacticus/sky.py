#! /usr/bin/env python

import sys,fnmatch
import numpy as np
import warnings
from .datasets import Dataset
from .constants import Pi
from .properties.manager import Property


def getRightAscension(X,Y,degrees=True):
    rightAscension = np.copy(np.arctan2(Y,X))
    mask = rightAscension < 0.0
    np.place(rightAscension,mask,2.0*Pi+rightAscension[mask])
    if degrees:
        rightAscension *= (180.0/Pi)
    return rightAscension


def getDeclination(X,Y,Z,degrees=True):
    R = np.sqrt(X**2+Y**2+Z**2)
    declination = np.copy(np.arcsin(Z/R))
    if degrees:
        declination *= (180.0/Pi)
    return declination



@Property.register_subclass('rightAscension')
class RightAscension(Property):

    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None,raiseError=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        match = propertyName == "rightAscension"
        if raiseError and not match:            
            msg = funcname+"(): Specified property '"+propertyName+\
                "' does not match 'rightAscension'."
            raise RuntimeError(msg)
        return match
    
    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        if not self.galaxies.GH5Obj.galaxyDatasetExists("lightconeRedshift",redshift):
            warnings.warn(funcname+"(): Unable to compute rightAscension as not a lightcone output.")
            return None
        # Create Dataset instance
        DATA = Dataset(name="rightAscension")
        # Extract necessary lightcone properties
        required = ["lightconePositionX","lightconePositionY"]
        GALS = self.galaxies.get(redshift,properties=required)
        X = GALS["lightconePositionX"].data
        Y = GALS["lightconePositionY"].data
        DATA.data = getRightAscension(X,Y,degrees=True)
        return DATA
    

@Property.register_subclass('declination')
class Declination(Property):

    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None,raiseError=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        match = propertyName == "declination"
        if raiseError and not match:            
            msg = funcname+"(): Specified property '"+propertyName+\
                "' does not match 'declination'."
            raise RuntimeError(msg)
        return match

    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        if not self.galaxies.GH5Obj.galaxyDatasetExists("lightconeRedshift",redshift):
            warnings.warn(funcname+"(): Unable to compute declination as not a lightcone output.")
            return None
        # Create Dataset instance
        DATA = Dataset(name="rightAscension")
        # Extract necessary lightcone properties
        required = ["lightconePositionX","lightconePositionY","lightconePositionZ"]
        GALS = self.galaxies.get(redshift,properties=required)
        X = GALS["lightconePositionX"].data
        Y = GALS["lightconePositionY"].data
        X = GALS["lightconePositionX"].data
        Z = GALS["lightconePositionZ"].data
        DATA.data = getDeclination(X,Y,Z,degrees=True)
        return DATA



