#! /usr/bin/env python

import sys
import re
import copy
import numpy as np
import scipy.interpolate
import h5py
import warnings
from .. import rcParams
from ..datasets import Dataset
from . import getEffectiveWavelength
from ..datasets import Dataset
from ..properties.manager import Property
from ..constants import metallicitySolar
from ..errors import ParseError

dustRegex = "(?P<dust>:dustCharlotFall2000)"


@Property.register_subclass('dustCharlotFallOpticalDepth')
class DustCharlotFallOpticalDepth(Property):
    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def parseStellarDatasetName(self,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check for stellar luminosity
        searchString = "^(?P<component>disk|spheroid)StellarOpticalDepth(?P<geometry>Clouds|ISM):"+\
            "(?P<filterName>[^:]+)(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+dustRegex+"$"
        MATCH = re.search(searchString,propertyName)
        return MATCH

    def parseLineDatasetName(self,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check for emission line luminosity
        searchString = "^(?P<component>disk|spheroid)LineOpticalDepth(?P<geometry>Clouds|ISM):"+\
            "(?P<lineName>[^:]+)(?P<filterName>:[^:]+)?"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            dustRegex+"$"
        MATCH = re.search(searchString,propertyName)
        return MATCH

    def parseDatasetName(self,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dustRegex = "(?P<dust>:dustCharlotFall2000)"
        # Check for stellar luminosity
        MATCH = self.parseStellarDatasetName(propertyName)
        if MATCH is not None:
            return MATCH
        # Check for emission line luminosity
        MATCH = self.parseLineDatasetName(propertyName)
        if MATCH is not None:
            return MATCH
        return None

    def matches(self,propertyName,redshift=None,raiseError=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid optical depth."
            raise RuntimeError(msg)
        return False
    
    def getOpticalDepthISM(self,redshift,component,effectiveWavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): Component must be 'disk' or 'spheroid'!")
        # Compute optical depth
        opticalDepth = component+"DustOpticalDepthCentral:dustAtlas"
        DATA = self.galaxies.get(redshift,properties=[opticalDepth])
        # Get ISM factors for Charlot & Fall model
        factorISM = rcParams.getfloat("dustCharlotFall","opticalDepthISMFactor",fallback=1.0)
        wavelengthZeroPoint = rcParams.getfloat("dustCharlotFall","wavelengthZeroPoint",fallback=5500.0)
        wavelengthExponent = rcParams.getfloat("dustCharlotFall","wavelengthExponent",fallback=0.7)
        wavelengthRatio = (effectiveWavelength/wavelengthZeroPoint)**wavelengthExponent
        # Compute ISM optical depth
        opticalDepthISM = factorISM*DATA[opticalDepth].data/wavelengthRatio
        return opticalDepthISM
        
    def getOpticalDepthClouds(self,redshift,component,effectiveWavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): Component must be 'disk' or 'spheroid'!")
        # Compute gas metallicity
        metalsName = component+"GasMetallicity"
        DATA = self.galaxies.get(redshift,properties=[metalsName])
        # Get ISM factors for Charlot & Fall model
        factorClouds = rcParams.getfloat("dustCharlotFall","opticalDepthCloudsFactor",fallback=1.0)
        wavelengthZeroPoint = rcParams.getfloat("dustCharlotFall","wavelengthZeroPoint",fallback=5500.0)
        wavelengthExponent = rcParams.getfloat("dustCharlotFall","wavelengthExponent",fallback=0.7)
        wavelengthRatio = (effectiveWavelength/wavelengthZeroPoint)**wavelengthExponent
        # Compute optical depth
        localISMMetallicity = rcParams.getfloat("dustOpticalDepth","localISMMetallicity",fallback=0.02)
        opticalDepthClouds = factorClouds*(DATA[metalsName].data*metallicitySolar)/localISMMetallicity
        opticalDepthClouds /= wavelengthRatio
        return opticalDepthClouds

    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,redshift=redshift,raiseError=True))
        DATA = Dataset(name=propertyName)
        DATA.attr["unitsInSI"] = 1.0
        # Parse property name
        MATCH = self.parseDatasetName(propertyName)
        # Get effective wavelength
        PROPS = self.galaxies.get(redshift,properties=["redshift"])
        wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)/1.0e4
        # Compute optical depth
        if MATCH.group('geometry') == "ISM":
            DATA.data = self.getOpticalDepthISM(redshift,MATCH.group('component'),wavelength)
        elif MATCH.group('geometry') == "Clouds":
            DATA.data = self.getOpticalDepthClouds(redshift,MATCH.group('component'),wavelength)
        return DATA


@Property.register_subclass('dustCharlotFall')
class DustCharlotFall(Property):

    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def parseStellarLuminosityDatasetName(self,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check for stellar luminosity
        searchString = "^(?P<component>disk|spheroid)LuminositiesStellar:"+\
            "(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            dustRegex+"(?P<recent>:recent)?$"
        MATCH = re.search(searchString,propertyName)
        return MATCH

    def parseLineLuminosityDatasetName(self,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check for emission line luminosity
        searchString = "^(?P<component>disk|spheroid)LineLuminosity:"+\
            "(?P<lineName>[^:]+)(?P<frame>:[^:]+)(?P<filterName>:[^:]+)?"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            dustRegex+"$"
        MATCH = re.search(searchString,propertyName)
        return MATCH


    def parseDatasetName(self,propertyName):
        """
        DustCharlotFall.parseDatasetName: Parse a dust parameters dataset.

        USAGE: SEARCH = DustCharlotFall.parseDatasetName(propertyName)

             INPUTS
                propertyName -- Property name to parse.

             OUTPUTS
                SEARCH       -- Regex search (re.search) object or None if
                                propertyName cannot be parsed.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dustRegex = "(?P<dust>:dustCharlotFall2000)"
        # Check for stellar luminosity
        MATCH = self.parseStellarLuminosityDatasetName(propertyName)
        if MATCH is not None:
            return MATCH
        # Check for emission line luminosity
        MATCH = self.parseLineLuminosityDatasetName(propertyName)
        if MATCH is not None:
            return MATCH
        return None


    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        DustCharlotFall.matches(): Returns boolean to indicate whether this class 
                                   can process the specified property.

        USAGE: match =  DustCharlotFall.matches(propertyName,[redshift=None])

          INPUTS 
             propertyName -- Name of property to process.
                 redshift -- Redshift value to query Galacticus HDF5 outputs.

          OUTPUTS 
                    match -- Boolean indicating whether this class can process 
                             this property.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid dust luminosity."
            raise RuntimeError(msg)
        return False
    
    def getOpticalDepthISM(self,redshift,component,effectiveWavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): Component must be 'disk' or 'spheroid'!")
        # Compute optical depth
        opticalDepth = component+"DustOpticalDepthCentral:dustAtlas"
        DATA = self.galaxies.get(redshift,properties=[opticalDepth])
        # Get ISM factors for Charlot & Fall model
        factorISM = rcParams.getfloat("dustCharlotFall","opticalDepthISMFactor",fallback=1.0)
        wavelengthZeroPoint = rcParams.getfloat("dustCharlotFall","wavelengthZeroPoint",fallback=5500.0)
        wavelengthExponent = rcParams.getfloat("dustCharlotFall","wavelengthExponent",fallback=0.7)
        wavelengthRatio = (effectiveWavelength/wavelengthZeroPoint)**wavelengthExponent
        # Compute ISM optical depth
        opticalDepthISM = factorISM*DATA[opticalDepth].data/wavelengthRatio
        return opticalDepthISM
        
    def getOpticalDepthClouds(self,redshift,component,effectiveWavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): Component must be 'disk' or 'spheroid'!")
        # Compute gas metallicity
        metalsName = component+"GasMetallicity"
        DATA = self.galaxies.get(redshift,properties=[metalsName])
        # Get ISM factors for Charlot & Fall model
        factorClouds = rcParams.getfloat("dustCharlotFall","opticalDepthCloudsFactor",fallback=1.0)
        wavelengthZeroPoint = rcParams.getfloat("dustCharlotFall","wavelengthZeroPoint",fallback=5500.0)
        wavelengthExponent = rcParams.getfloat("dustCharlotFall","wavelengthExponent",fallback=0.7)
        wavelengthRatio = (effectiveWavelength/wavelengthZeroPoint)**wavelengthExponent
        # Compute optical depth
        localISMMetallicity = rcParams.getfloat("dustOpticalDepth","localISMMetallicity",fallback=0.02)
        opticalDepthClouds = factorClouds*(DATA[metalsName].data*metallicitySolar)/localISMMetallicity
        opticalDepthClouds /= wavelengthRatio
        return opticalDepthClouds

    def attenuateStellarLuminosity(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,redshift=redshift,raiseError=True))
        if ":recent" in propertyName:
            raise RuntimeError(funcname+"(): Property '"+propertyName+"' cannot be a 'recent' stellar luminosity.")            
        # Check unattenuated and recent luminosities
        unattenuatedDatasetName = propertyName.replace(":dustCharlotFall2000","")
        recentDatasetName = propertyName.replace(":dustCharlotFall2000",":recent")
        PROPS = self.galaxies.get(redshift,properties=["redshift",unattenuatedDatasetName,
                                                       recentDatasetName])
        DATA = copy.copy(PROPS[unattenuatedDatasetName])
        DATA.name = propertyName
        if PROPS[recentDatasetName].data is None:
            DATA.data = None
            msg = funcname+"(): Cannot compute Charlot & Fall dust attenuation."
            msg = msg + " Unable to locate 'recent' luminosity '"+recentDatasetName+"'."
            warning.warn(msg)
            return DATA
        # Parse dataset name
        MATCH = self.parseStellarLuminosityDatasetName(propertyName)
        if MATCH is None:
            raise RuntimeError(funcname+"(): Property '"+propertyName+"' is not a stellar luminosity.")            
        # Get effective wavelength
        wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)/1.0e4
        # Extract optical depths
        opticalDepthISM = self.getOpticalDepthISM(redshift,MATCH.group('component'),wavelength)
        opticalDepthClouds = self.getOpticalDepthClouds(redshift,MATCH.group('component'),wavelength)
        # Compute attenuations
        attenuationISM = np.exp(-opticalDepthISM)
        attenuationClouds = np.exp(-opticalDepthClouds)
        # Attenuate luminosity
        attenuatedLuminosity = ((PROPS[unattenuatedDatasetName].data-PROPS[recentDatasetName].data) 
                                + PROPS[recentDatasetName].data*attenuationClouds)*attenuationISM
        DATA.data = attenuatedLuminosity
        return DATA

    def attenuateRecentStellarLuminosity(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,redshift=redshift,raiseError=True))
        if ":recent" not in propertyName:
            raise RuntimeError(funcname+"(): Property '"+propertyName+"' invalid. Must be a 'recent' stellar luminosity.")            
        # Check recent luminosity
        recentDatasetName = propertyName.replace(":dustCharlotFall2000","")
        PROPS = self.galaxies.get(redshift,properties=["redshift",recentDatasetName])
        DATA = copy.copy(PROPS[recentDatasetName])
        DATA.name = propertyName
        if PROPS[recentDatasetName].data is None:
            DATA.data = None
            msg = funcname+"(): Cannot compute Charlot & Fall dust attenuation."
            msg = msg + " Unable to locate 'recent' luminosity '"+recentDatasetName+"'."
            warning.warn(msg)
            return DATA
        # Parse dataset name
        MATCH = self.parseStellarLuminosityDatasetName(propertyName)
        if MATCH is None:
            raise RuntimeError(funcname+"(): Property '"+propertyName+"' is not a stellar luminosity.")            
        # Get effective wavelength
        wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)/1.0e4
        # Extract optical depths
        opticalDepthISM = self.getOpticalDepthISM(redshift,MATCH.group('component'),wavelength)
        opticalDepthClouds = self.getOpticalDepthClouds(redshift,MATCH.group('component'),wavelength)
        # Compute attenuations
        attenuationISM = np.exp(-opticalDepthISM)
        attenuationClouds = np.exp(-opticalDepthClouds)
        # Attenuate luminosity
        attenuatedLuminosity = PROPS[recentDatasetName].data*attenuationClouds*attenuationISM
        DATA.data = attenuatedLuminosity
        return DATA

    def attenuateLineLuminosity(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,redshift=redshift,raiseError=True))
        # Check unattenuated and recent luminosities
        unattenuatedDatasetName = propertyName.replace(":dustCharlotFall2000","")
        PROPS = self.galaxies.get(redshift,properties=["redshift",unattenuatedDatasetName])
        DATA = copy.copy(PROPS[unattenuatedDatasetName])
        DATA.name = propertyName
        # Parse dataset name
        MATCH = self.parseLineLuminosityDatasetName(propertyName)
        if MATCH is None:
            raise RuntimeError(funcname+"(): Property '"+propertyName+"' is not a line luminosity.")            
        # Get effective wavelength
        wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)/1.0e4
        # Extract optical depths
        opticalDepthISM = self.getOpticalDepthISM(redshift,MATCH.group('component'),wavelength)
        opticalDepthClouds = self.getOpticalDepthClouds(redshift,MATCH.group('component'),wavelength)
        # Compute attenuations
        attenuationISM = np.exp(-opticalDepthISM)
        attenuationClouds = np.exp(-opticalDepthClouds)
        # Attenuate luminosity
        attenuatedLuminosity = PROPS[unattenuatedDatasetName].data*attenuationClouds*attenuationISM
        DATA.data = attenuatedLuminosity
        return DATA

    def get(self,propertyName,redshift):
        """
        DustCharlotFall.get(): Compute dust-extinguished luminosities for 
                               specified redshift.

        USAGE:  DATA = DustCharlotFall.get(propertyName,redshift)

           INPUTS
                propertyName -- Name of property to compute.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUT 
                        DATA -- Instance of galacticus.datasets.Dataset() class 
                                containing computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,redshift=redshift,raiseError=True))
        if "LineLuminosity" in propertyName:
            DATA = self.attenuateLineLuminosity(propertyName,redshift)
        elif ":recent" in propertyName:
            DATA = self.attenuateRecentStellarLuminosity(propertyName,redshift)
        else:
            DATA = self.attenuateStellarLuminosity(propertyName,redshift)
        return DATA
