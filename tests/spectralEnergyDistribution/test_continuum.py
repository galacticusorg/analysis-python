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
from galacticus.spectralEnergyDistribution import parseDatasetName
from galacticus.spectralEnergyDistribution import getSpectralEnergyDistributionWavelengths
from galacticus.spectralEnergyDistribution.continuum import sedContinuum


class Test_sedContinuum(unittest.TestCase):

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
        self.SED = sedContinuum(GALS)
        return
            
    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.SED.galaxies.GH5Obj.close()
        del self.SED
        if self.removeExample:
            os.remove(self.snapshotFile)
        return
        
    def test_sedContinuumIdentifyTopHatLuminosityDatasets(self):
        # Check parse errors
        with self.assertRaises(ParseError):
            z = 1.000
            name = "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:z1.000"
            tophats = self.SED.identifyTopHatLuminosityDatasets(z,name)
        # Test recovery of top hat filters
        path = "galacticus.io.GalacticusHDF5.availableDatasets"
        with patch(path) as mocked_available:        
            wavelengths = np.linspace(1000.0,2000.0,100)
            names = ["spheroidLuminositiesStellar:adaptiveResolutionTopHat_"+str(w)+"_500:rest:z1.000" 
                     for w in wavelengths]
            names = names + ["diskLuminositiesStellar:adaptiveResolutionTopHat_"+str(w)+"_500:rest:z1.000" 
                             for w in wavelengths]
            names = names + ["diskMassStellar","spheroidMassStellar"]            
            mocked_available.return_value = names
            # Test extraction of disk top hats
            name = "diskSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000"
            tophats = self.SED.identifyTopHatLuminosityDatasets(z,name)
            disks = fnmatch.filter(names,"diskLuminositiesStellar:adaptiveResolutionTopHat*")
            self.assertEqual(tophats,disks)
            # Test extraction of spheroid top hats
            name = "spheroidSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000"
            tophats = self.SED.identifyTopHatLuminosityDatasets(z,name)
            spheroids = fnmatch.filter(names,"spheroidLuminositiesStellar:adaptiveResolutionTopHat*")
            self.assertEqual(tophats,spheroids)
            # Test extraction of total top hats
            name = "totalSpectralEnergyDistribution:1000.0_2000.0_100.0:rest:z1.000"
            tophats = self.SED.identifyTopHatLuminosityDatasets(z,name)
            totals = [filter.replace("disk","total") for filter in disks]
            self.assertEqual(tophats,totals)
        return

    def test_sedContinuumExtractTopHatWavelengths(self):
        # Test correct extraction of wavelengths of top hat filters
        wavelengths = np.linspace(900.0,2000.0,100)
        topHats = ["spheroidLuminositiesStellar:adaptiveResolutionTopHat_"+str(w)+"_500:rest:z1.000"
                   for w in wavelengths]
        topHats0 = copy.copy(topHats)
        waves,tHs = self.SED.extractTopHatWavelengths(topHats,sortTopHats=True)
        self.assertTrue(np.alltrue(wavelengths==waves))
        self.assertListEqual(topHats0,tHs)
        waves,tHs = self.SED.extractTopHatWavelengths(topHats,sortTopHats=False)
        self.assertTrue(np.alltrue(wavelengths==waves))
        self.assertListEqual(topHats0,tHs)
        # Test whether top hat filters are correctly sorted (if specified)
        shuffle(topHats)
        waves,tHs = self.SED.extractTopHatWavelengths(topHats,sortTopHats=False)
        self.assertTrue(np.alltrue(wavelengths==waves))
        self.assertListEqual(topHats,tHs)
        shuffle(topHats)
        waves,tHs = self.SED.extractTopHatWavelengths(topHats,sortTopHats=True)
        self.assertTrue(np.alltrue(wavelengths==waves))
        self.assertListEqual(tHs,topHats0)
        return

    def test_sedContinuumSelectWavelengthRange(self):
        # Test selection of correct wavelentth range
        wavelengths = np.linspace(900.0,2000.0,100)
        topHats = ["spheroidLuminositiesStellar:adaptiveResolutionTopHat_"+str(w)+"_500:rest:z1.000"
                   for w in wavelengths]
        topHats0 = copy.copy(topHats)
        # Check any errors are raised successfully
        with self.assertRaises(ValueError):
            self.SED.selectWavelengthRange(topHats,1700.0,1700.0)
            self.SED.selectWavelengthRange(topHats,1701.0,1700.0)
            self.SED.selectWavelengthRange(topHats,100.0,1200.0)
            self.SED.selectWavelengthRange(topHats,900.0,1200.0)
            self.SED.selectWavelengthRange(topHats,1200.0,2000.0)
            self.SED.selectWavelengthRange(topHats,1200.0,2001.0)
            self.SED.selectWavelengthRange(topHats,100.0,100.0)                        
        # Check wavelengths returned are in correct range
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Test: lower wavelength outside range
            lowerWavelength = 100.0
            upperWavelength = 1500.0
            mask = np.logical_and(wavelengths>=lowerWavelength,wavelengths<=upperWavelength)
            waves,tHs = self.SED.selectWavelengthRange(topHats,lowerWavelength,upperWavelength)
            self.assertEqual(waves.min(),wavelengths.min())
            idx = np.argwhere(mask).max()+1
            self.assertEqual(waves.max(),wavelengths[idx])
            mask[idx] = True
            self.assertListEqual(tHs,list(np.array(topHats0)[mask]))
            # Test: upper wavelength outside range
            lowerWavelength = 1200.0
            upperWavelength = 2001.0
            mask = np.logical_and(wavelengths>=lowerWavelength,wavelengths<=upperWavelength)
            waves,tHs = self.SED.selectWavelengthRange(topHats,lowerWavelength,upperWavelength)
            idx = np.argwhere(mask).min()-1
            self.assertEqual(waves.min(),wavelengths[idx])
            self.assertEqual(waves.max(),wavelengths.max())
            mask[idx] = True
            self.assertListEqual(tHs,list(np.array(topHats0)[mask]))
            # Test: both wavelengths inside range
            lowerWavelength = 1000.0
            upperWavelength = 1750.0
            mask = np.logical_and(wavelengths>=lowerWavelength,wavelengths<=upperWavelength)
            waves,tHs = self.SED.selectWavelengthRange(topHats,lowerWavelength,upperWavelength)
            idx = np.argwhere(mask).min()-1
            self.assertEqual(waves.min(),wavelengths[idx])
            mask[idx] = True
            idx = np.argwhere(mask).max()+1
            self.assertEqual(waves.max(),wavelengths[idx])
            mask[idx] = True
            self.assertListEqual(tHs,list(np.array(topHats0)[mask]))
            # Test: both wavelengths outside range
            lowerWavelength = 899.0
            upperWavelength = 2001.0
            mask = np.logical_and(wavelengths>=lowerWavelength,wavelengths<=upperWavelength)
            waves,tHs = self.SED.selectWavelengthRange(topHats,lowerWavelength,upperWavelength)
            self.assertEqual(waves.min(),wavelengths.min())
            self.assertEqual(waves.max(),wavelengths.max())
            self.assertListEqual(tHs,list(np.array(topHats0)[mask]))
        return

        
    def test_sedContinuumGetContinuumLuminosities(self):
        # Test extraction of top hat luminosities
        # First compute luminosities manually
        redshift = 1.0
        zStr = self.SED.galaxies.GH5Obj.getRedshiftString(redshift)
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr
        topHats = self.SED.identifyTopHatLuminosityDatasets(redshift,sedName)
        MATCH = parseDatasetName(sedName)
        lowerWavelength = float(MATCH.group("lowerWavelength"))
        upperWavelength = float(MATCH.group("upperWavelength"))
        wavelengths,topHats = self.SED.selectWavelengthRange(topHats,lowerWavelength,upperWavelength)
        TOPHATS = self.SED.galaxies.get(redshift,properties=topHats)
        luminosities = np.stack([TOPHATS[dset].data for dset in topHats],axis=1)
        # Test extraction using SED continuum function
        lums,waves = self.SED.getContinuumLuminosities(redshift,sedName)
        ngals = self.SED.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        self.assertEqual(lums.shape[0],ngals)
        self.assertEqual(lums.shape[1],len(waves))
        self.assertTrue(np.alltrue(lums==luminosities))
        self.assertTrue(np.alltrue(waves==wavelengths))        
        return
    
    def test_sedContinuumInterpolateContinuum(self):
        nwav = 100
        wavelengths = np.linspace(1000,2000,nwav)
        ngals = 200
        with self.assertRaises(ValueError):
            luminosities = np.random.rand(nwav*ngals).reshape(nwav,ngals)
            newLums = self.SED.interpolateContinuum(wavelengths,luminosities,wavelengths)
        luminosities = np.random.rand(nwav*ngals).reshape(ngals,nwav)
        newWaves = np.linspace(1200,1800,321)
        newLums = self.SED.interpolateContinuum(wavelengths,luminosities,newWaves)
        self.assertEqual(newLums.shape[0],ngals)
        self.assertEqual(newLums.shape[1],len(newWaves))
        return

    def test_sedContinuumAddContinuumNoise(self):
        nwav = 100
        ngals = 200
        wavelengths = np.linspace(1000,2000,nwav)
        luminosities = np.random.rand(nwav*ngals).reshape(ngals,nwav)
        # Test error raised for invalue S/N value
        with self.assertRaises(ValueError):
            continuum = self.SED.addContinuumNoise(wavelengths,luminosities,0.0)
            continuum = self.SED.addContinuumNoise(wavelengths,luminosities,-10.0)
        # Test array returned is correct shape
        continuum = self.SED.addContinuumNoise(wavelengths,luminosities,10.0)
        self.assertEqual(continuum.shape[0],ngals)
        self.assertEqual(continuum.shape[1],nwav)
        return
        
    def test_sedContinuumGet(self):
        # Test get continuum SED        
        redshift = 1.0
        ngals = self.SED.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        zStr = self.SED.galaxies.GH5Obj.getRedshiftString(redshift)
        # Extract continuum with no S/N specified        
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr
        luminosity0,wavelengths0 = self.SED.getContinuumLuminosities(redshift,sedName)        
        wavelengths = getSpectralEnergyDistributionWavelengths(sedName)
        luminosity = self.SED.interpolateContinuum(wavelengths0,luminosity0,wavelengths)
        waves,cont = self.SED.get(sedName,redshift)
        self.assertTrue(np.alltrue(wavelengths==waves))
        self.assertEqual(cont.shape[0],ngals)
        self.assertEqual(cont.shape[1],len(waves))                
        self.assertTrue(np.alltrue(luminosity==cont))
        # Extract continuum with non-zero S/N specified
        sedName = "diskSpectralEnergyDistribution:5000.0_8000.0_100.0:rest:"+zStr+":snr10.0"
        luminosity0,wavelengths0 = self.SED.getContinuumLuminosities(redshift,sedName)        
        wavelengths = getSpectralEnergyDistributionWavelengths(sedName)
        luminosity = self.SED.interpolateContinuum(wavelengths0,luminosity0,wavelengths)
        waves,cont = self.SED.get(sedName,redshift)
        self.assertTrue(np.alltrue(wavelengths==waves))
        self.assertEqual(cont.shape[0],ngals)
        self.assertEqual(cont.shape[1],len(waves))                
        self.assertFalse(np.alltrue(luminosity==cont))        
        return
        

if __name__ == "__main__":
    unittest.main()

