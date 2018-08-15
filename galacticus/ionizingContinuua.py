#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import unittest
from .datasets import Dataset
from .properties.manager import Property
from .filters.filters import GalacticusFilter
from .constants import luminosityAB,plancksConstant


@Property.register_subclass('ionizingContinuum')
class IonizingContinuum(Property):

    def __init__(self,galaxies):
        self.galaxies = galaxies
        # Set continuum units in photons/s
        self.continuumUnits = 1.0000000000000000e+50
        # Set filter names
        self.filterNames = {"Lyman":"Lyc","Helium":"HeliumContinuum","Oxygen":"OxygenContinuum"}
        return

    def parseDatasetName(self,datasetName):
        funcname = sys._getframe().f_code.co_name
        # Extract information from dataset name
        searchString = "^(?P<component>disk|spheroid)"+\
            "(?P<continuum>Lyman|Helium|Oxygen)ContinuumLuminosity"+\
            ":z(?P<redshift>[\d\.]+)(?P<recent>:recent)?$"
        return re.search(searchString,datasetName)

    def getConversionFactor(self,FILTER):
        conversion = (luminosityAB/plancksConstant/self.continuumUnits)
        mask = FILTER.transmission["transmission"] > 0.0
        minWavelength = FILTER.transmission["wavelength"][mask].min()
        maxWavelength = FILTER.transmission["wavelength"][mask].max()
        conversion *= np.log(maxWavelength/minWavelength)
        return conversion
    
    def matches(self,propertyName,redshift=None):
        if self.parseDatasetName(propertyName):
            return True
        return False

    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not an ionization contnuum luminosity."
            raise RuntimeError(msg)
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)
        # Extract appropriate stellar luminosity
        luminosityName = MATCH.group('component')+"LuminositiesStellar:"+\
            self.filterNames[MATCH.group('continuum')]+\
            ":rest:z"+MATCH.group('redshift')
        if MATCH.group('recent') is not None:
            luminosityName = luminosityName + MATCH.group('recent')
        GALS = self.galaxies.get(redshift,properties=[luminosityName])
        # Return None instance if stellar luminosity is missing
        if GALS[luminosityName] is None:
            return None
        # Load appropriate Galacticus filter
        FILTER = GalacticusFilter().load(self.filterNames[MATCH.group("continuum")])
        # Compute continuum luminosity
        DATA = Dataset(name=propertyName)
        DATA.data = np.copy(GALS[luminosityName].data)*self.getConversionFactor(FILTER)
        del GALS
        return DATA
        
class UnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from .galaxies import Galaxies
        from .io import GalacticusHDF5
        from .data import GalacticusData
        from shutil import copyfile
        # Locate the dynamic version of the galacticus.snapshotExample.hdf5 file.
        DATA = GalacticusData()
        self.snapshotFile = DATA.searchDynamic("galacticus.snapshotExample.hdf5")
        self.removeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.snapshotFile is None:
            self.snapshotFile = DATA.dynamic+"/examples/galacticus.snapshotExample.hdf5"
            self.removeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.snapshotExample.hdf5",self.snapshotFile)
        # Initialize the IonizingContinuum class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.ION = IonizingContinuum(GALS)
        # Create a list of incorrect names that will be rejected
        self._incorrectNames = [\
            "basicMass",\
                "totalOxygenContinuumLuminosity:z1.000",\
                "diskPlutoniumContinuumLuminosity:z1.000",\
                "diskOxygenContinuumLuminosity",\
                "diskOxygenContinuumLuminosity:1.000",\
                "diskOxygenContinuumLuminosityz1.000",\
                "diskOxygenContinuumLuminosity:recent"\
                ]
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.ION.galaxies.GH5Obj.close()
        del self.ION
        if self.removeExample:
            os.remove(self.snapshotFile)
        rcParams.restore()
        return

    def testFilterNames(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: IonizingContinuum: "+funcname)
        print("Testing IonizingContinuum continuum filter names attribute")
        self.assertEqual(len(self.ION.filterNames.keys()),3)
        self.assertEqual(self.ION.filterNames["Lyman"],"Lyc")
        self.assertEqual(self.ION.filterNames["Helium"],"HeliumContinuum")
        self.assertEqual(self.ION.filterNames["Oxygen"],"OxygenContinuum")
        print("TEST COMPLETE")
        print("\n")
        return
        
    def testConversionFactor(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: IonizingContinuum: "+funcname)
        print("Testing IonizingContinuum.getConversionFactor() function")        
        self.assertRaises(AttributeError,self.ION.getConversionFactor,None)
        for continuum in self.ION.filterNames.keys():            
            FILTER = GalacticusFilter().load(self.ION.filterNames[continuum])
            self.assertIsNotNone(FILTER)
            self.assertIsInstance(self.ION.getConversionFactor(FILTER),float)
            del FILTER.transmission
            self.assertRaises(AttributeError,self.ION.getConversionFactor,FILTER)        
        print("TEST COMPLETE")
        print("\n")
        return 

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: IonizingContinuum: "+funcname)
        print("Testing IonizingContinuum.matches() function")
        for component in ["disk","spheroid"]:
            for continuum in self.ION.filterNames.keys():
                name = component+continuum+"ContinuumLuminosity:z1.000"
                self.assertTrue(self.ION.matches(name))
                name = component+continuum+"ContinuumLuminosity:z1.000:recent"
                self.assertTrue(self.ION.matches(name))
        for name in self._incorrectNames:
            self.assertFalse(self.ION.matches(name))        
        print("TEST COMPLETE")
        print("\n")
        return

    def testParseDatasetName(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: IonizingContinuum: "+funcname)
        print("Testing IonizingContinuum.parseDatasetName() function")
        self.assertIsNone(self.ION.parseDatasetName("basicMass"))        
        for component in ["disk","spheroid"]:
            for continuum in self.ION.filterNames.keys():
                name = component+continuum+"ContinuumLuminosity:z1.000"
                MATCH = self.ION.parseDatasetName(name)                
                self.assertIsNotNone(MATCH)
                self.assertEqual(MATCH.group('component'),component)
                self.assertEqual(MATCH.group('continuum'),continuum)
                self.assertEqual(MATCH.group('redshift'),"1.000")
                self.assertIsNone(MATCH.group("recent"))
                name = component+continuum+"ContinuumLuminosity:z1.000:recent"
                MATCH = self.ION.parseDatasetName(name)                
                self.assertIsNotNone(MATCH)
                self.assertEqual(MATCH.group('component'),component)
                self.assertEqual(MATCH.group('continuum'),continuum)
                self.assertEqual(MATCH.group('redshift'),"1.000")
                self.assertEqual(MATCH.group("recent"),":recent")
        for name in self._incorrectNames:
            MATCH = self.ION.parseDatasetName(name)                
            self.assertIsNone(MATCH)
        print("TEST COMPLETE")
        print("\n")
        return
            
    def testGet(self):
        import warnings
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: IonizingContinuum: "+funcname)
        print("Testing IonizingContinuum.get() function")
        redshift = 1.0
        zStr = self.ION.galaxies.GH5Obj.getRedshiftString(redshift)
        for component in ["disk","spheroid"]:
            for continuum in self.ION.filterNames.keys():
                name = component+continuum+"ContinuumLuminosity:"+zStr
                DATA = self.ION.get(name,redshift)
                self.assertIsNotNone(DATA)
                self.assertEqual(DATA.name,name)
                self.assertIsNotNone(DATA.data)
                self.assertIsInstance(DATA.data,np.ndarray)
                name = component+continuum+"ContinuumLuminosity:z999.9999"
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore") 
                    self.assertIsNone(self.ION.get(name,redshift))
        for name in self._incorrectNames:
            self.assertRaises(RuntimeError,self.ION.get,name,redshift)
        print("TEST COMPLETE")
        print("\n")
        return
