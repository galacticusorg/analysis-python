#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import warnings
from ..datasets import Dataset
from ..properties.manager import Property
from ..Cloudy import CloudyTable
from ..constants import luminosityAB,erg,luminositySolar
from ..constants import centi,Pi,mega,parsec

def ergPerSecondPerCentimeterSquared(flux):
    flux = np.log10(flux)
    flux += np.log10(luminositySolar)
    flux -= np.log10(erg)
    flux -= np.log10((mega*parsec/centi)**2)
    flux = 10.0**flux
    return flux


@Property.register_subclass('emissionLineFlux')
class EmissionLineFlux(Property):
    
    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        self.CLOUDY = CloudyTable()
        return

    def lineInCloudyOutput(self,lineName):
        """
        EmissionLineFlux.lineInCloudyOutput: Returns boolean indicating whether specified
                                                   emission line can be found in CLOUDY output.

        USAGE:  result = EmissionLineFlux.lineInCloudyOutput(lineName)
        
          INPUTS
              lineName -- Name of emission line.

          OUTPUTS
              result   -- Boolean (T/F) indicating whether specified line is present.

        """
        return lineName in self.CLOUDY.listAvailableLines()

    def parseDatasetName(self,datasetName):
        """
        EmissionLineFlux.parseDatasetName: Parse an emission line luminosity 
                                                 dataset name.
        
        USAGE: SEARCH = EmissionLineFlux.parseDatasetName(propertyName)
        
             INPUTS
                propertyName -- Property name to parse.
                
             OUTPUTS
                SEARCH       -- Regex seearch (re.search) object or None if
                                propertyName cannot be parsed.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Construct search string to pass to regex
        searchString = "^(?P<component>disk|spheroid)LineFlux:"
        lines = "(?P<lineName>"+"|".join(self.CLOUDY.listAvailableLines())+")"
        searchString = searchString + lines + ":(?P<frame>rest|observed)"+\
            "(?P<filterName>:[^:]+)?"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            "(?P<recent>:recent)?(?P<dust>:dust[^:]+)?$"
        return re.search(searchString,datasetName)
    
    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        EmissionLineFlux.matches: Returns boolean to indicate whether this 
                                        class can process the specified property.

        USAGE:  match = EmissionLineFlux.matches(propertyName,[redshift=None],
                                                       [raiseError=False])
                
          INPUTS 
               propertyName -- Name of property to process.
                   redshift -- Redshift value to query Galacticus HDF5 outputs. 
                               (Redundant in this particular case, but required 
                               for other properties.)
                raiseError  -- Raise error if property does not match. 
                               (Default = False)                

          OUTPUTS 
                match       -- Boolean indicating whether this class can process 
                               this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid emission line flux. "+\
                "Available emission lines: "+\
                ", ".join(self.CLOUDY.listAvailableLines())+"."
            raise RuntimeError(msg)
        return False

    def get(self,propertyName,redshift):
        """
        EmissionLineFlux.get(): Compute specified emission line flux at specified
                                      redshift. 

        USAGE: DATA = EmissionLineFlux.get(propertyName,redshift)

           INPUTS
               propertyName -- Property name to compute flux for. Should be a
                               valid emission line flux dataset name.
               redshift     -- Redshift value to query Galacticus HDF5 outputs.
        
           OUTPUTS
               DATA         -- Dataset() class instance containing flux information, or
                               None if line flux cannot be computed.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)        
        # Extract line luminosity
        luminosityName = propertyName.replace("LineFlux","LineLuminosity")
        GALS = self.galaxies.get(redshift,properties=[luminosityName,"redshift"])
        # Check if line luminosity was calculated
        if GALS[luminosityName] is None:
            return None
        # Compute flux
        luminosityDistance = self.galaxies.GH5Obj.cosmology.luminosity_distance(GALS["redshift"].data)
        DATA = Dataset(name=propertyName)
        DATA.data = np.copy(GALS[luminosityName].data)/(4.0*Pi*luminosityDistance**2)
        attr = {"unitsInSI":luminositySolar/(mega*parsec)**2}
        attr["massHIIRegion"] = GALS[luminosityName].attr["massHIIRegion"]
        attr["lifetimeHIIRegion"] = GALS[luminosityName].attr["lifetimeHIIRegion"]
        del GALS
        del luminosityDistance
        return DATA


