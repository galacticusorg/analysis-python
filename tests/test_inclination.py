#! /usr/bin/env python

import sys,os
import numpy as np
import unittest
from shutil import copyfile
from galacticus import rcParams
from galacticus.datasets import Dataset
from galacticus.properties.manager import Property
from galacticus.constants import Pi
from galacticus.inclination import Generate_Random_Inclinations,Inclination
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData


class TestGenerateRandomInclinations(unittest.TestCase):
    
    def test_GenerateRandomInclinations(self):
        N = 100
        inc = Generate_Random_Inclinations(N,degrees=True)
        self.assertTrue(all(inc>=0.0))
        self.assertTrue(all(inc<=90.0))
        inc = Generate_Random_Inclinations(N,degrees=False)
        self.assertTrue(all(inc>=0.0))
        self.assertTrue(all(inc<=Pi/2.0))
        return

class TestInclination(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
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
        # Initialize the Inclination class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.INC = Inclination(GALS)
        return
        
    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.INC.galaxies.GH5Obj.close()
        del self.INC
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_InclinationMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.assertTrue(self.INC.matches("inclination"))
        self.assertFalse(self.INC.matches("redshift"))
        return

    def test_InclinationGet(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        redshift = 1.0
        DATA = self.INC.get("inclination",redshift)
        N = self.INC.galaxies.GH5Obj.countGalaxies(redshift)
        self.assertIsNotNone(DATA)
        self.assertEqual(DATA.name,"inclination")
        self.assertIsNotNone(DATA.data)
        self.assertEqual(DATA.data.size,N)
        self.assertTrue(len(DATA.attr.keys())>0)
        self.assertTrue("degrees" in DATA.attr.keys())
        with self.assertRaises(RuntimeError):
            self.INC.get("notAnInclination",redshift)
        return



if __name__ == "__main__":
    unittest.main()

