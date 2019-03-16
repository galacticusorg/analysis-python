#! /usr/bin/env python

import sys,re,glob,fnmatch
import numpy as np
import unittest
import warnings
from unittest.mock import patch
import copy
from random import shuffle
from galacticus.errors import ParseError
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.datasets import Dataset
from galacticus.spectralEnergyDistribution import parseDatasetName
from galacticus.spectralEnergyDistribution import getSpectralEnergyDistributionWavelengths
from galacticus.spectralEnergyDistribution.emissionLines import EmissionLines

class Test_sedEmissionLines(unittest.TestCase):

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
        # Initialize the SED continuum class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.LINES = EmissionLines(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.LINES.galaxies.GH5Obj.close()
        del self.LINES
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_sedEmissionLinesGetLineLuminosity(self):
        redshift = 1.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        ngals = self.LINES.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr
        MATCH = parseDatasetName(sedName)
        path = "galacticus.emissionLines.luminosities.EmissionLineLuminosity.get"
        luminosity = self.LINES.getLineLuminosity(MATCH,redshift,"balmerAlpha6563")
        self.assertEqual(len(luminosity),ngals)
        return

    def test_sedEmissionLinesAddLineProfile(self):
        redshift = 1.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        ngals = self.LINES.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr
        MATCH = parseDatasetName(sedName)
        wavelengths = getSpectralEnergyDistributionWavelengths(sedName)
        luminosities = np.zeros((ngals,len(wavelengths)),dtype=float)
        LINE = self.LINES.CLOUDY.lines["balmerAlpha6563"]
        self.LINES.addLineProfile(LINE,MATCH,redshift,wavelengths,luminosities)
        mask = np.invert(np.isnan(luminosities))
        self.assertTrue(np.any(luminosities[mask]>0.0))
        self.assertEqual(luminosities.shape[0],ngals)
        self.assertEqual(luminosities.shape[1],len(wavelengths))
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr+":fixedWidth100.0"
        MATCH = parseDatasetName(sedName)
        wavelengths = getSpectralEnergyDistributionWavelengths(sedName)
        luminosities = np.zeros((ngals,len(wavelengths)),dtype=float)
        LINE = self.LINES.CLOUDY.lines["balmerAlpha6563"]
        self.LINES.addLineProfile(LINE,MATCH,redshift,wavelengths,luminosities)
        mask = np.invert(np.isnan(luminosities))
        self.assertTrue(np.any(luminosities[mask]>0.0))
        self.assertEqual(luminosities.shape[0],ngals)
        self.assertEqual(luminosities.shape[1],len(wavelengths))
        return

    def test_sedEmissionLinesSumLineProfiles(self):
        redshift = 1.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        ngals = self.LINES.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr
        wavelengths = getSpectralEnergyDistributionWavelengths(sedName)
        luminosities = self.LINES.sumLineProfiles(sedName,redshift)
        mask = np.invert(np.isnan(luminosities))
        self.assertTrue(np.any(luminosities[mask]>0.0))
        self.assertEqual(luminosities.shape[0],ngals)
        self.assertEqual(luminosities.shape[1],len(wavelengths))
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr+":fixedWidth100.0"
        wavelengths = getSpectralEnergyDistributionWavelengths(sedName)
        luminosities = self.LINES.sumLineProfiles(sedName,redshift)
        mask = np.invert(np.isnan(luminosities))
        self.assertTrue(np.any(luminosities[mask]>0.0))
        self.assertEqual(luminosities.shape[0],ngals)
        self.assertEqual(luminosities.shape[1],len(wavelengths))
        return

    def test_sedEmissionLinesGet(self):
        redshift = 1.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        ngals = self.LINES.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr
        wavelengths = getSpectralEnergyDistributionWavelengths(sedName)
        luminosities = self.LINES.get(sedName,redshift)
        mask = np.invert(np.isnan(luminosities))
        self.assertTrue(np.any(luminosities[mask]>0.0))
        self.assertEqual(luminosities.shape[0],ngals)
        self.assertEqual(luminosities.shape[1],len(wavelengths))
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr+":fixedWidth100.0"
        wavelengths = getSpectralEnergyDistributionWavelengths(sedName)
        luminosities = self.LINES.get(sedName,redshift)
        mask = np.invert(np.isnan(luminosities))
        self.assertTrue(np.any(luminosities[mask]>0.0))
        self.assertEqual(luminosities.shape[0],ngals)
        self.assertEqual(luminosities.shape[1],len(wavelengths))
        return
        


        
if __name__ == "__main__":
    unittest.main()



