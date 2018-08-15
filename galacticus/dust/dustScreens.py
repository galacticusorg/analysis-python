#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import unittest
from scipy.interpolate import interp1d
from .. import rcParams
from ..datasets import Dataset
from ..Cloudy import CloudyTable
from ..filters.filters import GalacticusFilter
from ..properties.manager import Property
from .screens.manager import ScreenLaw

@Property.register_subclass('dustScreen')
class DustScreen(Property):

    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.SCREENS = ScreenLaw()
        self.CLOUDY = CloudyTable()
        self.GALFIL = GalacticusFilter()
        return

    def parseDatasetName(self,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dustRegex = "(?P<dust>:dust(?P<screen>"+"|".join(self.SCREENS.laws.keys())+\
            ")(_Av(?P<av>[\d\.]+))?)"
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
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid dust screen stellar or line luminosity. "
            msg = msgs + "Available dust screens: "+\
                " ,".join(self.SCREENS.laws.keys())+"."
            raise RuntimeError(msg)
        return False
        
    def getDustFreeName(self,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        return propertyName.replace(MATCH.group('dust'),"")
    
    def getDustFreeLuminosity(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dustFreeName = self.getDustFreeName(propertyName)
        GALS = self.galaxies.get(redshift,properties=[dustFreeName])
        return GALS[dustFreeName]

    def selectDustScreen(self,screen):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if screen not in self.SCREENS.laws.keys():
            msg = funcname+"(): Screen '"+screen+"' is not in list of available screen laws."
            msg = msgs + "Available dust screens: "+" ,".join(self.SCREENS.laws.keys())+"."
            raise KeyError(msg)
        return self.SCREENS.laws[screen]

    def getWavelength(self,propertyName):
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        if MATCH.group('filterName') is not None:
            FILTER = self.GALFIL.load(MATCH.group('filterName'))
            wavelength = FILTER.effectiveWavelength
        else:
            wavelength = self.CLOUDY.getWavelength(MATCH.group("lineName"))
        return wavelength

    def getAv(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        if MATCH.group("av") is None:
            name = MATCH.component('component')+"LuminositiesStellar"+\
                MATCH.group('redshiftString')+":dustCompendium:A_V"
            GALS = self.galaxies.get(redshift,properties=[name])
            AV = GALS[name].data
        else:
            N = self.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
            AV = np.ones(N,dtype=float)*float(MATCH.group('av'))
        return AV

    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        DATA = self.getDustFreeLuminosity(propertyName,redshift)
        # Return None if dust free luminosity cannot be found
        if DATA is None:
            return None        
        # Select dust screen to use
        SCREEN = self.selectDustScreen(MATCH.group('screen'))
        # Update dust free luminosity Dataset() with attenuated properties
        DATA.name = propertyName
        for key in SCREEN.attrs.keys():
            DATA.attr[key] = SCREEN.attrs[key]
        # Get wavelength at which to query dust screen
        wavelength = self.getWavelength(propertyName)
        # Get Av value
        Av = self.getAv(propertyName,redshift)
        # Compute attenuation
        atten = np.copy(SCREEN.curve(wavelength*angstrom/micron)*Av)
        del wavelength,Av
        # Attenuate luminosity
        DATA.data *= atten
        return DATA
            
    

