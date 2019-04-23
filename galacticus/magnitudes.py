#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import warnings
from . import rcParams
from .datasets import Dataset
from .properties.manager import Property
from .filters.filters import GalacticusFilter

@Property.register_subclass('magnitude')
class Magnitude(Property):

    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        self.GALFIL = GalacticusFilter()
        return

    @classmethod
    def parseDatasetName(cls,datasetName):
        """
        Magnitude.parseDatasetName: Parse a magnitude dataset name.

        USAGE: SEARCH = Magnitude.parseDatasetName(propertyName)

             INPUTS 
               propertyName -- Property name to parse.

             OUTPUTS 

                     SEARCH -- Regex seearch (re.search) object or
                               None if propertyName cannot be parsed.

        """
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Construct search string to pass to regex
        searchString = "^(?P<component>disk|spheroid|total)Magnitude(?P<magnitude>Apparent|Absolute):"+\
            "(?P<filter>[^:]+):(?P<frame>[^:]+)(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            "(?P<system>:vega|:AB)?(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
        return re.search(searchString,datasetName)

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        Function to identify whether this class can process a specified property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+"' is not a valid magnitude."
            raise RuntimeError(msg)
        return False

    @classmethod
    def getLuminosityName(cls,propertyName):
        """
        Given a magnitude dataset name, construct the appropriate luminosity dataset name.
        """
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = cls.parseDatasetName(propertyName)
        luminosityName = MATCH.group('component')+"LuminositiesStellar:"+MATCH.group("filter")+":"+\
            MATCH.group("frame")+MATCH.group("redshiftString")
        if MATCH.group("recent") is not None: luminosityName = luminosityName + MATCH.group("recent")
        if MATCH.group("dust") is not None: luminosityName = luminosityName + MATCH.group("dust")
        return luminosityName


    def getVegaOffset(self,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        # If None, then assume AB magnitude
        if MATCH.group("system") is None:
            return 0.0
        # AB magnitude
        if MATCH.group("system") == ":AB":
            return 0.0
        # If get to here then is a Vega magnitude
        FILTER = self.GALFIL.load(MATCH.group("filter"))
        return FILTER.vegaOffset


    def get(self,propertyName,redshift):
        """                                 
        Return a magnitude property.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        # Get luminosity dataset
        luminosityName = self.getLuminosityName(propertyName)
        # Read Galacticus properties
        properties = [luminosityName]
        if MATCH.group("magnitude") == "Apparent":
            properties.append("redshift")
        GALS = self.galaxies.get(redshift,properties=properties)
        if GALS[luminosityName] is None:
            return None
        # Create dataset
        DATA = Dataset(name=propertyName)
        # Compute absolute magnitude
        zeroCorrection = rcParams.getfloat("magnitude","zeroCorrection",fallback=1.0e-50)
        DATA.data = -2.5*np.log10(GALS[luminosityName].data+zeroCorrection)
        # Convert to Vega magnitudes if required        
        DATA.data += self.getVegaOffset(propertyName)
        # Convert to apparent magnitude if required
        if MATCH.group("magnitude") == "Apparent":
            distanceModulus = self.galaxies.GH5Obj.cosmology.band_corrected_distance_modulus(GALS["redshift"].data)
            DATA.data += distanceModulus
        return DATA
        
        
        
        
        
