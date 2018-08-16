#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import unittest
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
        return propertyName.replace(MATCH.group('dust'),"")
    
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

    def getWavelength(self,propertyName):
        """
        DustScreen.getWavelength(): For specified dust screen name, extract wavelength to 
                                    use for calculating dust attenuation.

        USAGE: wavelength = DustScreen.getWavelength(propertyName)
        
          INPUTS
            propertyName -- Name of dust attenuated dataset.
        
          OUTPUTS
            wavelength   -- Floating value for wavelength to use in dust screen.

        """
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        if MATCH.group('filterName') is not None:
            FILTER = self.GALFIL.load(MATCH.group('filterName').replace(":",""))
            wavelength = FILTER.effectiveWavelength
        else:
            wavelength = self.CLOUDY.getWavelength(MATCH.group("lineName"))
        return float(wavelength)

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
            name = MATCH.component('component')+"LuminositiesStellar"+\
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
        wavelength = self.getWavelength(propertyName)
        # Get Av value
        Av = self.getAv(propertyName,redshift)
        # Compute attenuation
        atten = np.copy(SCREEN.curve(wavelength*angstrom/micron)*Av)
        del wavelength,Av
        # Attenuate luminosity
        DATA.data *= atten
        return DATA
            
    

class UnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from ..galaxies import Galaxies
        from ..io import GalacticusHDF5
        from ..data import GalacticusData
        from shutil import copyfile
        # Locate the dynamic version of the galacticus.snapshotExample.hdf5 file.
        DATA = GalacticusData(verbose=False)
        self.snapshotFile = DATA.searchDynamic("galacticus.snapshotExample.hdf5")
        self.removeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.snapshotFile is None:
            self.snapshotFile = DATA.dynamic+"/examples/galacticus.snapshotExample.hdf5"
            self.removeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.snapshotExample.hdf5",self.snapshotFile)
        # Initialize the DustScreen class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.DUST = DustScreen(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DUST.galaxies.GH5Obj.close()
        del self.DUST
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def testListAvailableScreens(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Screens: "+funcname)
        print("Testing DustScreens.listAvailableScreens() function")
        self.assertGreater(len(self.DUST.listAvailableScreens()),0)
        print("TEST COMPLETE")
        print("\n")
        return

    def testParseDatasetName(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Parameters: "+funcname)
        print("Testing DustParameters.parseDatasetName() function")
        for screen in self.DUST.listAvailableScreens():
            name = "diskLuminositiesStellar:SDSS_r:observed:z1.000:dust"+screen
            MATCH = self.DUST.parseDatasetName(name)
            self.assertIsNotNone(MATCH)
            self.assertEqual(MATCH.group("screen"),screen)
            self.assertIsNone(MATCH.group("av"))
            name = "spheroidLineLuminosity:balmerAlpha6563:rest:SDSS_r:z1.000:dust"+screen+"_Av0.1"
            MATCH = self.DUST.parseDatasetName(name)
            self.assertIsNotNone(MATCH)
            self.assertEqual(MATCH.group("screen"),screen)
            self.assertEqual(MATCH.group("filterName"),":SDSS_r")
            self.assertEqual(MATCH.group("av"),"0.1")
            name = "spheroidLineLuminosity:balmerAlpha6563:rest:z1.000:dust"+screen
            MATCH = self.DUST.parseDatasetName(name)
            self.assertIsNotNone(MATCH)
            self.assertEqual(MATCH.group("screen"),screen)
            self.assertIsNone(MATCH.group("filterName"))
            name = "totalLineLuminositiy:balmerAlpha6563:rest:z1.000:"+screen
            MATCH = self.DUST.parseDatasetName(name)
            self.assertIsNone(MATCH)
        name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustClazeti"
        MATCH = self.DUST.parseDatasetName(name)
        self.assertIsNone(MATCH)
        name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCompendium"
        MATCH = self.DUST.parseDatasetName(name)
        self.assertIsNone(MATCH)
        name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti_Av"
        MATCH = self.DUST.parseDatasetName(name)
        self.assertIsNone(MATCH)
        print("TEST COMPLETE")
        print("\n")
        return

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Screens: "+funcname)
        print("Testing DustScreen.matches() function")
        for screen in self.DUST.listAvailableScreens():
            name = "diskLuminositiesStellar:SDSS_r:observed:z1.000:dust"+screen
            self.assertTrue(self.DUST.matches(name))
            name = "spheroidLineLuminosity:balmerAlpha6563:rest:SDSS_r:z1.000:dust"+screen+"_Av0.1"
            self.assertTrue(self.DUST.matches(name))
            name = "spheroidLineLuminosity:balmerAlpha6563:rest:z1.000:dust"+screen
            self.assertTrue(self.DUST.matches(name))
            name = "totalLineLuminositiy:balmerAlpha6563:rest:z1.000:dust"+screen
            self.assertFalse(self.DUST.matches(name,raiseError=False))
            self.assertRaises(RuntimeError,self.DUST.matches,name,raiseError=True)
            name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustClazeti"
            self.assertFalse(self.DUST.matches(name,raiseError=False))
            self.assertRaises(RuntimeError,self.DUST.matches,name,raiseError=True)
            name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCompendium"
            self.assertFalse(self.DUST.matches(name,raiseError=False))
            self.assertRaises(RuntimeError,self.DUST.matches,name,raiseError=True)
            name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti_Av"
            self.assertFalse(self.DUST.matches(name,raiseError=False))
            self.assertRaises(RuntimeError,self.DUST.matches,name,raiseError=True)
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetDustFreeName(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Screens: "+funcname)
        print("Testing DustScreen.getDustFreeName() function")
        name = "diskLuminositiesStellar:SDSS_r:rest:z1.000"
        self.assertEqual(self.DUST.getDustFreeName(name+":dustCalzetti"),name)
        self.assertEqual(self.DUST.getDustFreeName(name+":dustCalzetti_Av0.1"),name)
        self.assertRaises(RuntimeError,self.DUST.getDustFreeName,name+":Calzetti_Av")
        self.assertRaises(RuntimeError,self.DUST.getDustFreeName,name+":Clazeti")
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetDustFreeLuminosity(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Screens: "+funcname)
        print("Testing DustScreen.getDustFreeLuminosity() function")
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCalzetti"
        OUT = self.DUST.galaxies.GH5Obj.selectOutput(redshift)
        value = np.array(OUT["nodeData/diskLuminositiesStellar:SDSS_r:rest:"+zStr])
        DATA = self.DUST.getDustFreeLuminosity(name,redshift)
        self.assertIsNotNone(DATA)
        diff = np.fabs(DATA.data-value)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr
        OUT = self.DUST.galaxies.GH5Obj.selectOutput(redshift)
        value = self.DUST.galaxies.get(redshift,properties=[name])[name].data
        DATA = self.DUST.getDustFreeLuminosity(name+":dustCalzetti",redshift)
        self.assertIsNotNone(DATA)        
        diff = np.fabs(DATA.data-value)
        [self.assertLessEqual(d,1.0e-6) for d in diff if not np.isnan(d)]
        name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":Calzetti"
        self.assertRaises(RuntimeError,self.DUST.getDustFreeLuminosity,name,redshift)
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":Clazeti"
        self.assertRaises(RuntimeError,self.DUST.getDustFreeLuminosity,name,redshift)
        print("TEST COMPLETE")
        print("\n")
        return
        
    def testSelectDustScreen(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Screens: "+funcname)
        print("Testing DustScreen.selectDustScreen() function")
        for screen in self.DUST.listAvailableScreens():
            OBJ = self.DUST.selectDustScreen(screen)
            self.assertEqual(OBJ.__name__,screen)
        self.assertRaises(KeyError,self.DUST.selectDustScreen,"Clazeti")
        self.assertRaises(KeyError,self.DUST.selectDustScreen,"Compendium")        
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetWavelength(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Screens: "+funcname)
        print("Testing DustScreen.getWavelength() function")
        FILTER = self.DUST.GALFIL.load("SDSS_r")
        name = "diskLuminositiesStellar:SDSS_r:rest:z1.000:dustCalzetti"
        self.assertEqual(FILTER.effectiveWavelength,self.DUST.getWavelength(name))
        name = "diskLineLuminosity:balmerAlpha6563:rest:SDSS_r:z1.000:dustCalzetti"
        self.assertEqual(FILTER.effectiveWavelength,self.DUST.getWavelength(name))
        name = "diskLineLuminosity:balmerAlpha6563:rest:z1.000:dustCalzetti"
        self.assertEqual(self.DUST.CLOUDY.getWavelength("balmerAlpha6563"),\
                             self.DUST.getWavelength(name))
        name = "totalLineLuminositiy:balmerAlpha6563:rest:z1.000:dustCalzetti"
        self.assertRaises(RuntimeError,self.DUST.getWavelength,name)
        name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustClazeti"
        self.assertRaises(RuntimeError,self.DUST.getWavelength,name)
        name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCompendium"
        self.assertRaises(RuntimeError,self.DUST.getWavelength,name)
        name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti_Av"
        self.assertRaises(RuntimeError,self.DUST.getWavelength,name)
        print("TEST COMPLETE")
        print("\n")
        return
