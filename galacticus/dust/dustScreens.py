#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import unittest
from . import getEffectiveWavelength
from .screens.manager import ScreenLaw
from .. import rcParams
from ..datasets import Dataset
from ..Cloudy import CloudyTable
from ..filters.filters import GalacticusFilter
from ..properties.manager import Property
from ..constants import angstrom,micron

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

    def listAvailableScreens(self):
        """
        DustScreen.listAvailableScreens: Return a list of screen laws readily
                                         available.

        USAGE: screens = DustScreen.listAvailableScreens()

           OUTPUTS                
               screens -- List of available screens.

        """
        return self.SCREENS.laws.keys()

    def parseDatasetName(self,propertyName):
        """
        DustScreen.parseDatasetName: Parse a dust screen dataset.

        USAGE: SEARCH = DustScreen.parseDatasetName(propertyName)

             INPUTS
               propertyName -- Property name to parse.

             OUTPUTS                
               SEARCH       -- Regex search (re.search) object or 
                               None if propertyName cannot be parsed.

        """
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
        # Check for stellar SED-derived luminosity
        searchString = "^(?P<component>disk|spheroid)StellarSED:"+\
            "(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
            dustRegex+"$"
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
        DustScreen.matches: Returns boolean to indicate whether this class can process                                                                                    
                                 the specified property.

        USAGE: match = DustScreen.matches(propertyName,[redshift=None],[raiseError=None])

          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.  
              raiseError   -- Raise error if property does not match. (Default = False)

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
                "' is not a valid dust screen stellar or line luminosity. "
            msg = msg + "Available dust screens: "+\
                " ,".join(self.SCREENS.laws.keys())+"."
            raise RuntimeError(msg)
        return False
        
    def getDustFreeName(self,propertyName):
        """
        DustScreen.getDustFreeName(): Return the specified dataset name with the dust
                                      component removed (i.e. return the unattenuated
                                      dataset name).
        USAGE: name = DustScreen.getDustFreeName(propertyName)
        
          INPUTS
             propertyName -- Name of dust attenuated dataset.
        
          OUTPUTS
             name         -- Name of equivalent dust unattenuated dataset.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        dustFreeName = propertyName.replace(MATCH.group('dust'),"")
        if re.search("StellarSED",dustFreeName):
            dustFreeName = dustFreeName.replace(MATCH.group('frame'),"")
        return dustFreeName
    
    def getDustFreeLuminosity(self,propertyName,redshift):
        """
        DustScreen.getDustLuminosity(): For dust attenuated dataset, Return the 
                                        corresponding unattenuated luminosity.

        USAGE: DATA = DustScreen.getDustFreeLuminosity(propertyName,redshift)
        
          INPUTS
             propertyName -- Name of dust attenuated dataset.
             redshift     -- Redshift value to query Galacticus HDF5 file.
        
          OUTPUTS
             DATA         -- Dataset() instance for unattenuated luminosity.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dustFreeName = self.getDustFreeName(propertyName)
        GALS = self.galaxies.get(redshift,properties=[dustFreeName])
        return GALS[dustFreeName]

    def selectDustScreen(self,screen):
        """
        DustScreen.selectDustScreen(): Return class for specified dust screen name.

        USAGE: OBJ = DustScreen.selectDustScreen(screen)
        
          INPUTS
             screen -- Name of dust screen class to extract. E.g. Calzetti, Allen, ...
        
          OUTPUTS
             OBJ    -- Instance of class object for specified dust screen.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if screen not in self.SCREENS.laws.keys():
            msg = funcname+"(): Screen '"+screen+"' is not in list of available screen laws."
            msg = msg + "Available dust screens: "+" ,".join(self.SCREENS.laws.keys())+"."
            raise KeyError(msg)
        return self.SCREENS.laws[screen]

    def getAv(self,propertyName,redshift):
        """
        DustScreen.getAv(): Return V-band attenuation parameter.

        USAGE: AV = DustScreen.getAv(propertyName,redshift)
        
          INPUTS
             propertyName -- Name of dust attenuated dataset.
             redshift     -- Redshift value to query Galacticus HDF5 file.
        
          OUTPUTS
             AV           -- Numpy array of V-band attenuations.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        if MATCH.group("av") is None:
            name = MATCH.group('component')+"LuminositiesStellar"+\
                MATCH.group('redshiftString')+":dustCompendium:A_V"
            GALS = self.galaxies.get(redshift,properties=[name])
            AV = GALS[name].data
        else:
            N = self.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
            AV = np.ones(N,dtype=float)*float(MATCH.group('av'))
        return AV

    def get(self,propertyName,redshift):
        """
        DustScreen.get(): Compute dust attenuated luminosity.

        USAGE: DATA = DustScreen.get(propertyName,redshift)
        
          INPUTS
             propertyName -- Name of dust attenuated dataset.
             redshift     -- Redshift value to query Galacticus HDF5 file.
        
          OUTPUTS
             DATA         -- Dataset() class instance containing attenuated
                             luminosity information, or None instance if
                             attenuated luminosity could not be computed.

        """
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
        PROPS = self.galaxies.get(redshift,properties=["redshift"])
        wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)
        # Get Av value
        Av = self.getAv(propertyName,redshift)
        # Compute attenuation
        atten = np.copy(SCREEN.curve(wavelength*angstrom/micron)*Av)
        del wavelength,Av
        # Attenuate luminosity
        atten = np.minimum(10.0**(-0.4*atten),1.0)
        DATA.data *= atten
        return DATA
            
    
