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
from galacticus.redshift import Redshift
from galacticus.constants import speedOfLight

class TestRedshift(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Locate the dynamic version of the example Galacticus HDF5 files
        DATA = GalacticusData()
        self.snapshotFile = DATA.searchDynamic("galacticus.snapshotExample.hdf5")
        self.lightconeFile = DATA.searchDynamic("galacticus.lightconeExample.hdf5")
        self.removeSnapshotExample = False
        self.removeLightconeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.snapshotFile is None:
            self.snapshotFile = DATA.dynamic+"/examples/galacticus.snapshotExample.hdf5"
            self.removeSnapshotExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.snapshotExample.hdf5",self.lightconeFile)
        if self.lightconeFile is None:
            self.lightconeFile = DATA.dynamic+"/examples/galacticus.lightconeExample.hdf5"
            self.removeLightconeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.lightconeExample.hdf5",self.lightconeFile)
        # Initialize the Redshift class.
        SGALS = Galaxies(GH5Obj=GalacticusHDF5(self.snapshotFile,'r'))
        self.SZ = Redshift(SGALS)
        LGALS = Galaxies(GH5Obj=GalacticusHDF5(self.lightconeFile,'r'))
        self.LZ = Redshift(LGALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.SZ.galaxies.GH5Obj.close()
        self.LZ.galaxies.GH5Obj.close()
        del self.SZ 
        del self.LZ
        if self.removeSnapshotExample:
            os.remove(self.snapshotFile)
        if self.removeLightconeExample:
            os.remove(self.lightconeFile)
        return

    def test_RedshiftMatches(self):
        z = 1.0
        # Check available options
        for name in self.SZ.availableOptions:
            self.assertTrue(self.SZ.matches(name,redshift=z,raiseError=False))
        # Check bad names
        names = ["redshiftSnapshot","totalMassStellar","lightconeRedshift"]
        for name in names:
            self.assertFalse(self.SZ.matches(name,redshift=z,raiseError=False))
            with self.assertRaises(RuntimeError):
                M = self.SZ.matches(name,redshift=z,raiseError=True)
        return

    def test_RedshiftGetObservedRedshift(self):
        z = 1.0
        # Check instance trying to process snapshot output
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.assertIsNone(self.SZ.getObservedRedshift(z))
        # Check instance processing lightcone output
        DATA = self.LZ.getObservedRedshift(z)
        self.assertEqual(DATA.name,"observedRedshift")
        required = ["lightconeRedshift","lightconePositionX","lightconePositionY","lightconePositionZ",\
                        "lightconeVelocityX","lightconeVelocityY","lightconeVelocityZ"]
        GALS = self.LZ.galaxies.get(z,properties=required)
        X = GALS["lightconePositionX"].data
        Y = GALS["lightconePositionY"].data
        Z = GALS["lightconePositionZ"].data
        VX = GALS["lightconeVelocityX"].data
        VY = GALS["lightconeVelocityY"].data
        VZ = GALS["lightconeVelocityZ"].data
        zCos = GALS["lightconeRedshift"].data
        R = np.sqrt(X**2+Y**2+Z**2)
        v_r = (VX*X + VY*Y + VZ*Z)/R
        c_kms = speedOfLight/1000.0
        zobs = np.copy((1.0+zCos)*(1.0+v_r/c_kms)-1.0)
        self.assertTrue(np.array_equal(DATA.data,zobs))
        return

    def test_RedshiftGetSnapshotRedshift(self):
        z = 1.0
        # Check snapshot instance
        zsnap = np.ones(self.SZ.galaxies.GH5Obj.countGalaxiesAtRedshift(z),
                        dtype=float)*self.SZ.galaxies.GH5Obj.nearestRedshift(z)
        DATA = self.SZ.getSnapshotRedshift(z)
        self.assertEqual(DATA.name,"snapshotRedshift")
        self.assertTrue(np.array_equal(DATA.data,zsnap))
        # Check lightcone instance
        zsnap = np.ones(self.LZ.galaxies.GH5Obj.countGalaxiesAtRedshift(z),
                        dtype=float)*self.LZ.galaxies.GH5Obj.nearestRedshift(z)
        DATA = self.LZ.getSnapshotRedshift(z)
        self.assertEqual(DATA.name,"snapshotRedshift")
        self.assertTrue(np.array_equal(DATA.data,zsnap))
        return

    def test_RedshiftGetRedshift(self):
        z = 1.0
        # Check snapshot isntance
        DATA = self.SZ.getRedshift(z)
        DATA2 = self.SZ.getSnapshotRedshift(z)
        self.assertEqual(DATA.name,"redshift")
        self.assertTrue(np.array_equal(DATA.data,DATA2.data))
        # Check lightcone instance
        DATA = self.LZ.getRedshift(z)
        OUT = self.LZ.galaxies.GH5Obj.selectOutput(z)
        ztrue = np.array(OUT["nodeData/lightconeRedshift"])
        self.assertEqual(DATA.name,"redshift")
        self.assertTrue(np.array_equal(DATA.data,ztrue))
        return

    def test_RedshiftGet(self):
        z = 1.0
        # Check raises error for incorrect property names
        with self.assertRaises(RuntimeError):
            self.SZ.get("lightconeRedshift",z)
            self.LZ.get("lightconeRedshift",z)
        # Check snapshot instances
        DATA = self.SZ.get("snapshotRedshift",z)
        DATA2 = self.SZ.getSnapshotRedshift(z)
        self.assertEqual(DATA.name,DATA2.name)
        self.assertTrue(np.array_equal(DATA.data,DATA2.data))
        DATA = self.SZ.get("redshift",z)
        self.assertEqual(DATA.name,"redshift")
        self.assertTrue(np.array_equal(DATA.data,DATA2.data))                        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.assertIsNone(self.SZ.get("observedRedshift",z))
        # Check lightcone instances
        DATA = self.LZ.get("snapshotRedshift",z)
        DATA2 = self.LZ.getSnapshotRedshift(z)
        self.assertEqual(DATA.name,DATA2.name)
        self.assertTrue(np.array_equal(DATA.data,DATA2.data))
        DATA = self.LZ.get("observedRedshift",z)
        DATA2 = self.LZ.getObservedRedshift(z)
        self.assertEqual(DATA.name,DATA2.name)
        self.assertTrue(np.array_equal(DATA.data,DATA2.data))
        DATA = self.LZ.get("redshift",z)
        OUT = self.LZ.galaxies.GH5Obj.selectOutput(z)
        ztrue = np.array(OUT["nodeData/lightconeRedshift"])
        self.assertEqual(DATA.name,"redshift")
        self.assertTrue(np.array_equal(DATA.data,ztrue))
        return

                         


if __name__ == "__main__":
    unittest.main()

