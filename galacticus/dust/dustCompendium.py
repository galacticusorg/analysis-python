#! /usr/bin/env python

import sys
import re
import copy
import numpy as np
import scipy.interpolate
import h5py
import warnings
from .CompendiumTable import CompendiumTable
from . import getEffectiveWavelength
from .. import rcParams
from ..datasets import Dataset
from ..properties.manager import Property
from ..constants import megaParsec, massSolar, centi, milli
from ..filters import Filter
from ..filters.filters import GalacticusFilter
from ..data import GalacticusData
from ..Cloudy import CloudyTable


COMPENDIUM = CompendiumTable()
CLOUDY = CloudyTable()
FILTERS = GalacticusFilter()

@Property.register_subclass('dustCompendium')
class DustCompendium(Property):
    """
    DustCompendium: Compute dust-extinguished luminosities using the dust compendium tabulations.

    Functions:
            matches(): Indicates whether specified dataset can be processed by this class.
            get(): Computes dust-extinguished luminosities at specified redshift.

    """    
    dustCompendiumRegEx = "^(?P<component>disk|spheroid)LuminositiesStellar:(?P<filter>[^:]+):(?P<frame>[^:]+)"+\
                          "(?P<redshiftString>:z(?P<redshift>[\d\.]+)):dustCompendium"

    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies     = galaxies
        self.data         = GalacticusData(verbose=False)
        self.tablesLoaded = False
        return

    def parseDatasetName(self,propertyName):
        """
        DustCompendium.parseDatasetName: Parse a dust parameters dataset.

        USAGE: SEARCH = DustCompendium.parseDatasetName(propertyName)

             INPUTS 
                propertyName -- Property name to parse.

             OUTPUTS
                SEARCH       -- Regex search (re.search) object or None if
                                propertyName cannot be parsed.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dustRegex = "(?P<dust>:dustCompendium)"
        # Check for stellar luminosity
        searchString = "^(?P<component>disk|spheroid)LuminositiesStellar:"+\
            "(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            dustRegex+"(?P<recent>:recent)?$"
        MATCH = re.search(searchString,propertyName)
        if MATCH is not None:
            return MATCH
        # Check for emission line luminosity
        searchString = "^(?P<component>disk|spheroid)LineLuminosity:"+\
            "(?P<lineName>[^:]+)(?P<frame>:[^:]+)(?P<filterName>:[^:]+)?"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            dustRegex+"(?P<recent>:recent)?$"
        MATCH = re.search(searchString,propertyName)
        if MATCH is not None:
            return MATCH
        return None

    
    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        DustCompendium.matches(): Returns boolean to indicate whether this class can process
                                 the specified property.

        USAGE: match =  DustCompendium.matches(propertyName,[redshift=None])

          INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.                           

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
                "' is not a valid dust compendium property."
            raise RuntimeError(msg)
        return False

    def get(self,propertyName,redshift):        
        """
        DustCompendium.get(): Compute dust-extinguished luminosities for specified redshift.
        
        USAGE:  DATA = DustCompendium.get(propertyName,redshift)
                
           INPUTS
           
                propertyName -- Name of property to compute.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.
           
           OUTPUT
                DATA         -- Instance of galacticus.datasets.Dataset() class containing 
                                computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,redshift=redshift,raiseError=True))
        # Parse dataset name
        MATCH = self.parseDatasetName(propertyName)
        # Extract properties needed to compute attenuation
        unattenuatedDatasetName = propertyName.replace(":dustCompendium","")
        properties = [unattenuatedDatasetName,"diskDustOpticalDepthCentral:dustCompendium",
                      "inclination","redshift","diskRadius"]
        if MATCH.group('component') == "spheroid":
            properties.append("spheroidRadius")
        PROPS = self.galaxies.get(redshift,properties=properties)
        # Get effective wavelength (convert from angstroms to microns)
        wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)/1.0e4
        # Create mask to avoid missing galaxies
        opticalDepthMask = np.invert(np.isnan(PROPS["diskDustOpticalDepthCentral:dustCompendium"].data))
        if MATCH.group('component') == "spheroid":
            opticalDepthMask = np.logical_and(opticalDepthMask,PROPS["spheroidRadius"].data>0.0)
            spheroidScaleRadius = np.ones_like(PROPS["spheroidRadius"].data)*np.nan
            spheroidScaleRadius[opticalDepthMask] = \
                PROPS["spheroidRadius"].data[opticalDepthMask]/PROPS["diskRadius"].data[opticalDepthMask]
        # Interpolate over Compendium table            
        if MATCH.group('component') == "spheroid":
            attenuations = COMPENDIUM.getSpheroidAttenuation(wavelength,
                                                             PROPS["inclination"].data,
                                                             spheroidScaleRadius,
                                                             PROPS["diskDustOpticalDepthCentral:dustCompendium"].data,
                                                             opticalDepthMask=opticalDepthMask)
        else:
            attenuations = COMPENDIUM.getDiskAttenuation(wavelength,
                                                         PROPS["inclination"].data,
                                                         PROPS["diskDustOpticalDepthCentral:dustCompendium"].data,
                                                         opticalDepthMask=opticalDepthMask)
        # Raise warnings for any attenuations greater than unity
        if any(attenuations>1.0):
            msg = funcname+"(): Some of the computed attenuations are greater than unity. "+\
                "Setting upper limit of unity."
            warnings.warn(msg)
            attenuations = np.minimum(attenuations,1.0)
        # Apply attenuation to unattenuated luminosity and return Dataset object
        DATA = Dataset(name=propertyName)
        DATA.attr = copy.copy(PROPS[unattenuatedDatasetName].attr)
        DATA.data = np.copy(PROPS[unattenuatedDatasetName].data)*attenuations
        return DATA

