#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import unittest
from . import rcParams
from .datasets import Dataset
from .properties.manager import Property
from .constants import metallicitySolar,mega,massSolar,parsec


@Property.register_subclass('metallicity')
class Metallicity(Property):    
    """
    Metallicity: Compute galaxy metallicities.

    Functions: 
        matches(): Indicates whether specified dataset can be
                   processed by this class.  
        get(): Computes galaxy metallicty at specified redshift.
        parseDatasetName(): Parse the dataset name using regex.

    """
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def parseDatasetName(self,datasetName):
        """
        Metallicity.parseDatasetName(): Parse the specified dataset name using regex. Will
                                        return a re.search() instance if datasetName is
                                        '(disk|spheroid|total)Metallicity'. Otherwise will
                                        return a 'None' instance.
        
        USAGE:  SEARCH =  Metallicity.parseDatasetName(datasetName)

            INPUTS
                datasetName -- Dataset name to parse.

            OUTPUTS
                SEARCH      -- A re.search instance or None if datasetName is not a valid
                               metallicty dataset name.
              
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        searchString = "^(?P<component>disk|spheroid|total)(?P<phase>Gas|Stellar)Metallicity$"
        return re.search(searchString,datasetName)

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        Metallicity.matches(): Returns boolean to indicate whether this class can process
                               the specified property.

        USAGE: match = Metallicty.matches(propertyName,[redshift=None])

           INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. (Redundant
                              in this particular case, but required for other properties.)

           OUTPUTS
              match        -- Boolean indicating whether this class can process
                              this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid metallictiy dataset name. "+\
                "Syntax is: (disk|spheroid|total)(Gas|Stellar)Metallicity."
            raise RuntimeError(msg)
        return False

    def get(self,propertyName,redshift):
        """
        Metallicity.get(): Compute galaxy metallicities for specified redshift.

        USAGE: DATA = Metallicity.get(propertyName,redshift)

           INPUTS
                propertyName -- Name of property to compute. This should be set 
                                to '(disk|spheroid|total)(Gas|Stellar)Metallicity'.  
                redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUT 
                DATA         -- Instance of galacticus.datasets.Dataset() class 
                                containing computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a metallicity."
            raise RuntimeError(msg)
        # Extract  mass and metals abundance
        MATCH = self.parseDatasetName(propertyName)
        massName = MATCH.group('component')+"Mass"+MATCH.group('phase')
        metalsName = MATCH.group('component')+"Abundances"+MATCH.group('phase')+"Metals"
        GALS = self.galaxies.get(redshift,properties=[massName,metalsName])
        # Extract abdunances and remove any negative values
        abundance = np.copy(GALS[metalsName].data)
        mask = abundance < 0.0
        if any(mask):
            abundance[mask] = 0.0
        # Extract gas mass
        mass = np.copy(GALS[massName].data)
        # Convert any values with zero gas mass to avoid divide by zero
        metallicity = np.zeros_like(mass)
        mask = mass>0.0
        metallicity[mask] = np.copy(abundance[mask]/mass[mask])
        # Clear GALS from memory
        del GALS,mass,abundance
        # Compute metallicity
        DATA = Dataset(name=propertyName)
        DATA.data = np.copy(metallicity)
        DATA.data /= metallicitySolar        
        del metallicity
        # Apply zero offset correction
        zeroCorrection = rcParams.getfloat("metals","zeroCorrection",fallback=1.0e-50)
        DATA.data += zeroCorrection
        return DATA



@Property.register_subclass('metalsGasDensity')
class MetalsGasDensity(Property):

    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        return

    @classmethod
    def parseDatasetName(cls,datasetName):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Construct search string to pass to regex
        searchString = "^(?P<component>disk|spheroid)MetalsGasDensity$"
        return re.search(searchString,datasetName)
    
    @classmethod
    def matches(cls,propertyName,redshift=None,raiseError=False):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = cls.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid hydrogen gas density. "+\
                "Syntax is (disk|spheroid)MetalsGasDensity."
            raise RuntimeError(msg)
        return False

    def getSurfaceDensityMetals(self,component,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): requires either a 'disk' or 'spheroid' component.")
        # Extract metal mass and galaxy radius
        metals = component+"MassGas"
        radius = component+"Radius"
        GALS = self.galaxies.get(redshift,properties=[metals,radius])
        # Compute surface density in pc**2
        area = Pi*np.copy(mega*GALS[radius].data)**2
        densitySurfaceMetals = np.zeros_like(GALS[radius].data)
        mask = area>0.0
        densitySurfaceMetals[mask] = np.copy(GALS[metals].data[mask]/area[mask])
        # Select method for computing density (central or mass-weighted)
        method = rcParams.get("hydrogenGasDensity","densityMethod",fallback="central")
        if method.lower() == "central":
            densitySurfaceGas /= 2.0
        elif method.lower() == "massweighted":
            densitySurfaceGas /= 8.0
        else:
            msg = funcname+"(): in rcParams hydrogenGasDensty/densityMethod "+\
                "should be either 'central' of 'massWeighted'. Default=central."
            raise ValueError(msg)
        return densitySurfaceGas

    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)
        component = MATCH.group('component')
        # Create dataset
        DATA = Dataset(name=propertyName)
        attr = {"unitsInSI":massSolar/(mega*parsec)**2}
        DATA.attr = attr
        DATA.data = np.copy(self.getSurfaceDensityMetals(component,redshift))
        # Apply zero offset correction
        zeroCorrection = rcParams.getfloat("metals","zeroCorrection",fallback=1.0e-50)
        DATA.data += zeroCorrection
        return DATA



