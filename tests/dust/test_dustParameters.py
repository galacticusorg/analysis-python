#! /usr/bin/env python

import os
import numpy as np
import unittest
from shutil import copyfile
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.dust.dustParameters import DustParameters


class TestDustParameters(unittest.TestCase):
        
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
        # Initialize the DustCompendium class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.DUST = DustParameters(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DUST.galaxies.GH5Obj.close()
        del self.DUST
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_DustParametersParseDatasetName(self):
        for component in ["disk","spheroid","total"]:
            for dust in ["Atlas","Compendium","CharlotFall2000","Calzetti"]:
                name = component+"LuminositiesStellar:z1.000:dust"+dust+":A_V"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = component+"LuminositiesStellar:z1.000:dust"+dust+":R_V"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
        names = ["diskLuminositiesStellar:z1.000:dustCompendium",
                 "diskLuminositiesStellar:z1.000:dustAltas:A_V",
                 "basicMass"]
        for name in names:
            self.assertIsNone(self.DUST.parseDatasetName(name))
        return

    def test_DustParametersMatches(self):
        for component in ["disk","spheroid","total"]:
            for dust in ["Atlas","Compendium","CharlotFall2000","Calzetti"]:
                name = component+"LuminositiesStellar:z1.000:dust"+dust+":A_V"
                self.assertTrue(self.DUST.matches(name))
                name = component+"LuminositiesStellar:z1.000:dust"+dust+":R_V"
                self.assertTrue(self.DUST.matches(name))
        names = ["diskLuminositiesStellar:z1.000:dustCompendium",
                 "diskLuminositiesStellar:z1.000:dustAltas:A_V",
                 "basicMass"]
        for name in names:
            self.assertFalse(self.DUST.matches(name))
            with self.assertRaises(RuntimeError):
                self.DUST.matches(name,raiseError=True)
        return

    def test_DustParametersGetAttenuationParameter(self):        
        N = 100
        attenL = np.random.rand(N)
        unattenL = np.ones_like(attenL)
        result = self.DUST.getAttenuationParameter(attenL,unattenL)
        self.assertEqual(result.shape,attenL.shape)
        with self.assertRaises(ValueError):
            self.DUST.getAttenuationParameter(unattenL,1.0)
        return

    def test_DustParametersGetReddeningParameter(self):        
        N = 100
        attenL = np.random.rand(N)
        unattenL = np.ones_like(attenL)
        result = self.DUST.getReddeningParameter(attenL,unattenL,attenL,unattenL)
        self.assertEqual(result.shape,attenL.shape)
        with self.assertRaises(ValueError):
            self.DUST.getReddeningParameter(unattenL,1.0,1.0,1.0)
        return

    
    def test_DustParametersGet(self):
        names = ["diskLuminositiesStellar:z1.000:dustCompendium",
                 "diskLuminositiesStellar:z1.000:dustAltas:A_V",
                 "basicMass"]
        for name in names:
            with self.assertRaises(RuntimeError):
                self.DUST.get(name,1.0)
        for component in ["disk","spheroid","total"]:
            for dust in ["Compendium","Calzetti"]:
                redshift = 1.0
                zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
                name = component+"LuminositiesStellar:"+zStr+":dust"+dust+":A_V"
                result = self.DUST.get(name,1.0)
                self.assertEqual(result.name,name)
                self.assertIsNotNone(result.data)
                name = component+"LuminositiesStellar:"+zStr+":dust"+dust+":R_V"
                result = self.DUST.get(name,1.0)
                self.assertEqual(result.name,name)
                self.assertIsNotNone(result.data)
        return

if __name__ == "__main__":
    unittest.main()


