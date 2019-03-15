#! /usr/bin/env python

import sys,re,glob,fnmatch,os
import numpy as np
import unittest
import warnings
from unittest.mock import patch
import copy
from random import shuffle
from galacticus import rcParams
from galacticus.errors import ParseError
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.datasets import Dataset
from galacticus.constants import megaParsec,centi,jansky,erg,luminosityAB,Pi
from galacticus.spectralEnergyDistribution import parseDatasetName
from galacticus.spectralEnergyDistribution.continuum import sedContinuum
from galacticus.spectralEnergyDistribution.emissionLines import sedEmissionLines
from galacticus.spectralEnergyDistribution.spectralEnergyDistribution import SpectralEnergyDistribution


class Test_SpectralEnergyDistribution(unittest.TestCase):

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
        # Initialize the SED class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.SED = SpectralEnergyDistribution(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.SED.galaxies.GH5Obj.close()
        del self.SED
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_SpectralEnergyDistributionMatches(self):        
        # Test matches function for SpectralEnergyDistribution class for various names options
        names = [
            "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000",
            "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:observed:z1.000:fixedWidth100.0",
            "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000:snr10.0",
            "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000:dustCompendium",
            "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000:noLines",
            "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:observed:z1.000:fixedWidth100.0:snr10.0",
            "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000:snr10.0:noLines",
            "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000:snr10.0:dustCompendium:noLines"
            ]
        for name in names:
            self.assertTrue(self.SED.matches(name))
        names = [
            "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:z1.000",
            "diskSpectralEnergyDistribution:1000.0_2000.0:observed:z1.000",
            "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:observed:z1.000:width100.0",
            "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:snr10.0:z1.000",
            "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000:sn10.0",
            "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000:dustCompendium:snr10.0",
            "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:observed:fixedWidth100.0:snr10.0:z1.000",
            "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000:noLines:snr10.0",
            "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000::dustCompendium:snr10.0:noLines"            
            ]
        for name in names:
            self.assertFalse(self.SED.matches(name))
            with self.assertRaises(RuntimeError):
                self.SED.matches(name,raiseError=True)
        return
    
    def test_SpectralEnergyDistributionErgPerSecond(self):
        # Test function for converting SED units to erg/s.
        sed0 = np.random.rand(100).reshape(20,5)        
        zeroCorrection = rcParams.getfloat("spectralEnergyDistribution",
                                           "zeroCorrection",
                                           fallback=1.0e-50)        
        sedT = np.log10(np.copy(sed0)+zeroCorrection)
        sedT += np.log10(luminosityAB)
        sedT -= np.log10(erg)
        sedT = 10.0**sedT
        sedC = self.SED.ergPerSecond(sed0)
        diff = np.fabs(sedT-sedC).flatten()
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        return


    def test_SpectralEnergyDistributionConvertToMicroJanskies(self):
        # Test function for converting SED units to micro-Janskies.
        redshift = 1.0
        z = self.SED.galaxies.get(redshift,properties=["redshift"])["redshift"].data
        sed0 = np.repeat(np.random.rand(len(z)),5).reshape((len(z),5))
        sedT = self.SED.ergPerSecond(np.copy(sed0))
        comDistance = self.SED.galaxies.GH5Obj.cosmology.comoving_distance(z)*megaParsec/centi
        comDistance = np.repeat(comDistance,sedT.shape[1]).reshape(sedT.shape)
        sedT /= 4.0*Pi*comDistance**2
        sedT /= jansky
        sedT *= 1.0e6
        sedC = self.SED.convertToMicroJanskies(redshift,np.copy(sed0))
        diff = np.fabs(sedT-sedC).flatten()
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        return

    def test_SpectralEnergyDistributionGet(self):
        # Test the 'get' function for SpectralEnergyDistribution class
        redshift = 1.0
        zStr = self.SED.galaxies.GH5Obj.getRedshiftString(redshift)
        # Test instance with emission lines
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr
        LINES = sedEmissionLines(self.SED.galaxies)
        CONT = sedContinuum(self.SED.galaxies)
        wave,C = CONT.get(sedName,redshift)
        L = LINES.get(sedName,redshift)
        sedTrue = self.SED.convertToMicroJanskies(redshift,C+L)
        DATA = self.SED.get(sedName,redshift)
        diff = np.fabs(DATA.data-sedTrue).flatten()
        [self.assertLessEqual(d,1.0e-6) for d in diff if d is not np.nan]        
        # Test instance with no emission lines
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr+":noLines"
        LINES = sedEmissionLines(self.SED.galaxies)
        CONT = sedContinuum(self.SED.galaxies)
        wave,C = CONT.get(sedName,redshift)
        sedTrue = self.SED.convertToMicroJanskies(redshift,C)
        DATA = self.SED.get(sedName,redshift)
        diff = np.fabs(DATA.data-sedTrue).flatten()
        [self.assertLessEqual(d,1.0e-6) for d in diff if d is not np.nan]        
        return



if __name__ == "__main__":
    unittest.main()

