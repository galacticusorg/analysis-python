#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from .datasets import Dataset
from .properties.manager import Property
from .filters.filters import GalacticusFilter
from .constants import luminosityAB,plancksConstant

def parseConinuumLuminosity(datasetName):
    funcname = sys._getframe().f_code.co_name
    # Extract information from dataset name
    searchString = "^(?P<component>disk|spheroid)"+\
        "(?P<continuum>Lyman|Helium|Oxygen)ContinuumLuminosity"+\
        ":z(?P<redshift>[\d\.]+)(?P<recent>:recent)?$"
    return re.search(searchString,datasetName)


@Property.register_subclass('ionizingContinuum')
class IonizingContinuum(Property):

    def __init__(self,galaxies):
        self.galaxies = galaxies
        # Set continuum units in photons/s
        self.continuumUnits = 1.0000000000000000e+50
        # Set filter names
        self.filterNames = {"Lyman":"Lyc","Helium":"HeliumContinuum","Oxygen":"OxygenContinuum"}
        return

    def getConversionFactor(self,FILTER):
        conversion = (luminosityAB/plancksConstant/self.continuumUnits)
        mask = FILTER.transmission["transmission"] > 0.0
        minWavelength = FILTER.transmission["wavelength"][mask].min()
        maxWavelength = FILTER.transmission["wavelength"][mask].max()
        conversion *= np.log(maxWavelength/minWavelength)
        return conversion
    
    
    def matches(self,propertyName,redshift=None):
        if parseConinuumLuminosity(propertyName):
            return True
        return False

    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not an ionization contnuum luminosity."
            raise RuntimeError(msg)
        # Extract information from property name
        MATCH = parseConinuumLuminosity(propertyName)
        # Extract appropriate stellar luminosity
        luminosityName = MATCH.group('component')+"LuminositiesStellar:"+\
            self.filterNames[MATCH.group('continuum')]+\
            ":rest:z"+MATCH.group('redshift')
        if MATCH.group('recent') is not None:
            luminosityName = luminosityName + MATCH.group('recent')
        GALS = self.galaxies(redshift,properties=[luminosityName])
        # Return None instance if stellar luminosity is missing
        if GALS[luminosityName] is None:
            return None
        # Load appropriate Galacticus filter
        FILTER = GalacticusFilter().load(self.filterNames[MATCH.group("continuum")]+".xml")
        # Compute continuum luminosity
        DATA = Dataset(name=propertyName)
        DATA.data = np.copy(GALS[luminosityName].data)*self.getConversionFactor(FILTER)
        del GALS
        return DATA
        


        
        
        
