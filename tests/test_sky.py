#! /usr/bin/env python

import sys,os
import fnmatch
import numpy as np
import unittest
import warnings
from shutil import copyfile
from galacticus import rcParams
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.constants import Pi
from galacticus.sky import RightAscension,Declination,getRightAscension,getDeclination


class TestRightAscension(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Locate the dynamic version of the example Galacticus HDF5 files
        DATA = GalacticusData()
        self.lightconeFile = DATA.searchDynamic("galacticus.lightconeExample.hdf5")
        self.removeLightconeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.lightconeFile is None:
            self.lightconeFile = DATA.dynamic+"/examples/galacticus.lightconeExample.hdf5"
            self.removeLightconeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.lightconeExample.hdf5",self.lightconeFile)
        # Initialize the Redshift class.
        GALS = Galaxies(GH5Obj=GalacticusHDF5(self.lightconeFile,'r'))
        self.RA = RightAscension(GALS)
        return
    
    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.RA.galaxies.GH5Obj.close()
        del self.RA
        if self.removeLightconeExample:
            os.remove(self.lightconeFile)
        return

    def test_RightAscensionMatches(self):
        z = 1.0
        # Check correct option
        self.assertTrue(self.RA.matches("rightAscension",redshift=z,raiseError=False))
        # Check incorrect options
        names = ['basicMass','declination','ra','rightascension','right_ascension']
        for name in names:
            self.assertFalse(self.RA.matches(name,redshift=z,raiseError=False))
            with self.assertRaises(RuntimeError):
                M = self.RA.matches(name,redshift=z,raiseError=True)
        return

    def test_getRightAscension(self):    
        X =  np.array([1.0, 1.0,  0.0,  -1.0,  -1.0, -1.0,  0.0,  1.0])
        Y =  np.array([0.0, 1.0,  1.0,   1.0,   0.0, -1.0, -1.0, -1.0])
        ra = np.arange(len(X))*(Pi/4.0)
        self.assertTrue(np.allclose(ra,getRightAscension(X,Y,degrees=False)))
        ra *= (180.0/Pi)
        self.assertTrue(np.allclose(ra,getRightAscension(X,Y,degrees=True)))
        return

    def test_RightAscensionGet(self):
        z = 1.0
        # Check raises error for incorrect property names
        with self.assertRaises(RuntimeError):
            self.RA.get("declination",z)
            self.RA.get("RightAscension",z)            
            self.RA.get("right_ascension",z)
        # Check returns correct values for lightcone output
        GALS = self.RA.galaxies.get(z,properties=["lightconePositionX","lightconePositionY"])
        ra = getRightAscension(GALS["lightconePositionX"].data,GALS["lightconePositionY"].data,degrees=True)
        DATA = self.RA.get("rightAscension",z)
        self.assertTrue(DATA.name,"rightAscension")
        self.assertTrue(np.allclose(DATA.data,ra))
        return


class TestDeclination(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Locate the dynamic version of the example Galacticus HDF5 files
        DATA = GalacticusData()
        self.lightconeFile = DATA.searchDynamic("galacticus.lightconeExample.hdf5")
        self.removeLightconeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.lightconeFile is None:
            self.lightconeFile = DATA.dynamic+"/examples/galacticus.lightconeExample.hdf5"
            self.removeLightconeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.lightconeExample.hdf5",self.lightconeFile)
        # Initialize the Redshift class.
        GALS = Galaxies(GH5Obj=GalacticusHDF5(self.lightconeFile,'r'))
        self.DEC = Declination(GALS)
        return
    
    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DEC.galaxies.GH5Obj.close()
        del self.DEC
        if self.removeLightconeExample:
            os.remove(self.lightconeFile)
        return

    def test_DeclinationMatches(self):
        z = 1.0
        # Check correct option
        self.assertTrue(self.DEC.matches("declination",redshift=z,raiseError=False))
        # Check incorrect options
        names = ['basicMass','dec','Declination','rightAscension']
        for name in names:
            self.assertFalse(self.DEC.matches(name,redshift=z,raiseError=False))
            with self.assertRaises(RuntimeError):
                M = self.DEC.matches(name,redshift=z,raiseError=True)
        return

    def test_getDeclination(self):    
        X = np.array([0.0,1.0,1.0, 1.0, 0.0])
        Y = np.array([0.0,0.0,0.0, 0.0, 0.0])
        Z = np.array([1.0,1.0,0.0,-1.0,-1.0])
        dec = (Pi/2.0) - np.arange(len(X))*(Pi/4.0)
        self.assertTrue(np.allclose(dec,getDeclination(X,Y,Z,degrees=False)))
        dec *= (180.0/Pi)
        self.assertTrue(np.allclose(dec,getDeclination(X,Y,Z,degrees=True)))
        return

    def test_DeclinationGet(self):
        z = 1.0
        # Check raises error for incorrect property names
        with self.assertRaises(RuntimeError):
            self.DEC.get("dec",z)
            self.DEC.get("Declination",z)            
            self.DEC.get("right_ascension",z)
        # Check returns correct values for lightcone output
        xName = "lightconePositionX"
        yName = "lightconePositionY"
        zName = "lightconePositionZ"
        GALS = self.DEC.galaxies.get(z,properties=[xName,yName,zName])
        ra = getDeclination(GALS[xName].data,GALS[yName].data,GALS[zName].data,degrees=True)
        DATA = self.DEC.get("declination",z)
        self.assertTrue(DATA.name,"declination")
        self.assertTrue(np.allclose(DATA.data,ra))
        return
        
    


if __name__ == "__main__":
    unittest.main()

