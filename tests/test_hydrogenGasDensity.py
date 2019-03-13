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
from galacticus.constants import parsec,massSolar
from galacticus.constants import mega,centi,Pi
from galacticus.constants import massAtomic,atomicMassHydrogen,massFractionHydrogen
from galacticus.hydrogenGasDensity import HydrogenGasDensity

class TestHydrogenGasDensity(unittest.TestCase):
    

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
        # Initialize the hydrogenGasDensity
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.DENS = HydrogenGasDensity(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DENS.galaxies.GH5Obj.close()
        del self.DENS
        if self.removeExample:
            os.remove(self.snapshotFile)
        return
        
    def test_hydrogenGasDensityParseDatasetName(self):
        for comp in ["disk","spheroid"]:
            name = comp + "HydrogenGasDensity"
            M = self.DENS.parseDatasetName(name)
            self.assertTrue(M is not None)
            self.assertEqual(M.group('component'),comp)
        for name in ["HydrogenGasDensity",
                     "diskGasDensity",
                     "totalHydrogenGasDensity"]:
            M = self.DENS.parseDatasetName(name)
            self.assertIsNone(M)
        return

    def test_hydrogenGasDensityMatches(self):
        for comp in ["disk","spheroid"]:
            name = comp + "HydrogenGasDensity"
            self.assertTrue(self.DENS.matches(name))
        for name in ["HydrogenGasDensity",
                     "diskGasDensity",
                     "totalHydrogenGasDensity"]:
            self.assertFalse(self.DENS.matches(name,raiseError=False))
            with self.assertRaises(RuntimeError):
                self.DENS.matches(name,raiseError=True)                             
        return

    def test_hydrogenGasDensityGetMassGiantMolecularClouds(self):
        value = 1.0e4
        rcParams.set("hydrogenGasDensity","massGMC",str(value))
        self.assertEqual(value,self.DENS.getMassGiantMolecularClouds())
        rcParams.remove_option("hydrogenGasDensity","massGMC")
        self.assertEqual(self.DENS.getMassGiantMolecularClouds(),3.7e7)
        return
            
    def test_hydrogenGasDensityGetCriticalSurfaceDensityClouds(self):
        value = 1.0e4
        rcParams.set("hydrogenGasDensity","surfaceDensityCritical",str(value))
        self.assertEqual(self.DENS.getCriticalSurfaceDensityClouds(),value)
        rcParams.remove_option("hydrogenGasDensity","surfaceDensityCritical")
        self.assertEqual(self.DENS.getCriticalSurfaceDensityClouds(),8.5e13)
        return
    
    def test_hydrogenGasDensityGetSurfaceDensityGas(self):
        z = 1.0
        # Check will fail for 'total' component
        with self.assertRaises(ValueError):
            self.DENS.getSurfaceDensityGas("total",z)
        # Test retrieval of correct values
        component = "disk"
        for method in ["central","massWeighted"]:
            rcParams.set("hydrogenGasDensity","densityMethod",method)
            gas = component+"MassGas"
            radius = component+"Radius"
            GALS = self.DENS.galaxies.get(z,properties=[gas,radius])
            area = Pi*np.copy(GALS[radius].data)**2
            np.place(area,area==0.0,np.nan)
            densitySurfaceGas = GALS[gas].data/area
            np.place(densitySurfaceGas,densitySurfaceGas==0.0,np.nan)
            if method.lower() == "central":
                densitySurfaceGas /= 2.0
            elif method.lower() == "massweighted":
                densitySurfaceGas /= 8.0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                data = self.DENS.getSurfaceDensityGas(component,z)
            diff = np.fabs(data-densitySurfaceGas)
            [self.assertLessEqual(d,1.0e-6) for d in diff if np.invert(np.isnan(d))]
            maskD = np.isnan(data)
            mask0 = np.isnan(densitySurfaceGas)
            [self.assertEqual(m,n) for m,n in zip(maskD,mask0)]
        rcParams.set("hydrogenGasDensity","densityMethod","unknown")
        with self.assertRaises(ValueError):
            data = self.DENS.getSurfaceDensityGas("disk",z)
        return

    def test_hydrogenGasDensityGet(self):
        z = 1.0
        # Check returns Runtime error for bad names
        for name in ["HydrogenGasDensity",
                     "diskGasDensity",
                     "totalHydrogenGasDensity"]:
            with self.assertRaises(RuntimeError):
                self.DENS.get(name,z)                
        # Check values
        component = "disk"
        rcParams.set("hydrogenGasDensity","densityMethod","central")
        rcParams.set("hydrogenGasDensity","massGMC","3.7e7")
        rcParams.set("hydrogenGasDensity","surfaceDensityCritical","8.5e13")
        densitySurfaceGas = self.DENS.getSurfaceDensityGas(component,z)
        massGMC = self.DENS.getMassGiantMolecularClouds()
        surfaceDensityCritical = self.DENS.getCriticalSurfaceDensityClouds()
        massClouds = massGMC/(densitySurfaceGas/surfaceDensityCritical)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            densitySurfaceClouds = np.maximum(densitySurfaceGas,surfaceDensityCritical)
        densityHydrogen = (3.0/4.0)*np.sqrt(Pi)/np.sqrt(massClouds)
        densityHydrogen *= densitySurfaceClouds**1.5
        densityHydrogen *= (centi/(mega*parsec))**3
        densityHydrogen *= massFractionHydrogen*massSolar
        densityHydrogen /= (massAtomic*atomicMassHydrogen)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            DATA = self.DENS.get(component+"HydrogenGasDensity",z)
        self.assertEqual(component+"HydrogenGasDensity",DATA.name)
        diff = np.fabs(DATA.data-densityHydrogen)
        [self.assertLessEqual(d,1.0e-6) for d in diff if np.invert(np.isnan(d))]
        maskD = np.isnan(DATA.data)
        mask0 = np.isnan(densityHydrogen)
        [self.assertEqual(m,n) for m,n in zip(maskD,mask0)]
        return
    



if __name__ == "__main__":
    unittest.main()

