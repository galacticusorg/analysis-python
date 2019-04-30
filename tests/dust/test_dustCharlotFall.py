#! /usr/bin/env python

import os
import numpy as np
import warnings
import unittest
from shutil import copyfile
from galacticus import rcParams
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.dust.dustCharlotFall import DustCharlotFall
from galacticus.dust import getEffectiveWavelength
from galacticus.constants import metallicitySolar


class TestDustCharlotFall(unittest.TestCase):

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
        self.DUST = DustCharlotFall(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DUST.galaxies.GH5Obj.close()
        del self.DUST
        if self.removeExample:
            os.remove(self.snapshotFile)
        return    

    def test_DustCharlotFallParseStellarLuminosityDatasetName(self):
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000"
        self.assertIsNotNone(self.DUST.parseStellarLuminosityDatasetName(name))
        name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000:recent"
        self.assertIsNotNone(self.DUST.parseStellarLuminosityDatasetName(name))
        name = "totalLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000"
        self.assertIsNone(self.DUST.parseStellarLuminosityDatasetName(name))
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000"
        self.assertIsNone(self.DUST.parseStellarLuminosityDatasetName(name))        
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000:recent"
        self.assertIsNone(self.DUST.parseStellarLuminosityDatasetName(name))        
        return

    def test_DustCharlotFallParseLineLuminosityDatasetName(self):
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000"
        self.assertIsNotNone(self.DUST.parseLineLuminosityDatasetName(name))
        name = "totalLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000"
        self.assertIsNone(self.DUST.parseLineLuminosityDatasetName(name))        
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000:recent"
        self.assertIsNone(self.DUST.parseLineLuminosityDatasetName(name))        
        name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000"
        self.assertIsNone(self.DUST.parseLineLuminosityDatasetName(name))        
        name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000:recent"
        self.assertIsNone(self.DUST.parseLineLuminosityDatasetName(name))        
        return

    def test_DustCharlotFallParseDatasetName(self):        
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        for component in ["disk","spheroid"]:
            for frame in ["rest","observed"]:
                name = component+"LuminositiesStellar:SDSS_r:"+frame+":"+zStr+":dustCharlotFall2000"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = name + ":recent"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = component+"LineLuminosity:balmerAlpha6563:"+frame+":"+zStr+":dustCharlotFall2000"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = component+"LineLuminosity:balmerAlpha6563:"+frame+":SDSS_r:"+zStr+":dustCharlotFall2000"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
        name = "totalLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000"
        self.assertIsNone(self.DUST.parseDatasetName(name))
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000:recent"        
        self.assertIsNone(self.DUST.parseDatasetName(name))
        name = "diskLineLuminosity:balmerAlpha6563:rest:SDSS_r:"+zStr+":dustCharlotFall2000:recent"        
        self.assertIsNone(self.DUST.parseDatasetName(name))
        name = "basicMass"
        self.assertIsNone(self.DUST.parseDatasetName(name))
        return

    def test_DustCharlotFallMatches(self):
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        for component in ["disk","spheroid"]:
            for frame in ["rest","observed"]:
                name = component+"LuminositiesStellar:SDSS_r:"+frame+":"+zStr+":dustCharlotFall2000"
                self.assertTrue(self.DUST.matches(name))
                name = name + ":recent"
                self.assertTrue(self.DUST.matches(name))
                name = component+"LineLuminosity:balmerAlpha6563:"+frame+":"+zStr+":dustCharlotFall2000"
                self.assertTrue(self.DUST.matches(name))
                name = component+"LineLuminosity:balmerAlpha6563:"+frame+":SDSS_r:"+zStr+":dustCharlotFall2000"
                self.assertTrue(self.DUST.matches(name))
        names = ["totalLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000",
                 "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000:recent",
                 "diskLineLuminosity:balmerAlpha6563:rest:SDSS_r:"+zStr+":dustCharlotFall2000:recent",
                 "basicMass"]            
        for name in names:
            self.assertFalse(self.DUST.matches(name))
            with self.assertRaises(RuntimeError):
                self.DUST.matches(name,raiseError=True)
        return

    def test_DustCharlotFallGetOpticalDepthISM(self):
        z = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(z)
        effWave = 0.6563
        # Check returns error for incorrect component
        with self.assertRaises(ValueError):
            self.DUST.getOpticalDepthISM(z,"total",effWave)
        # Check calculation
        opticalDepth = "diskDustOpticalDepthCentral:dustAtlas"
        DATA = self.DUST.galaxies.get(z,properties=[opticalDepth])
        factorISM = rcParams.getfloat("dustCharlotFall","opticalDepthISMFactor",fallback=1.0)
        wavelengthZeroPoint = rcParams.getfloat("dustCharlotFall","wavelengthZeroPoint",fallback=5500.0)
        wavelengthExponent = rcParams.getfloat("dustCharlotFall","wavelengthExponent",fallback=0.7)
        wavelengthRatio = (effWave/wavelengthZeroPoint)**wavelengthExponent
        opticalDepthISM = factorISM*DATA[opticalDepth].data/wavelengthRatio        
        result = self.DUST.getOpticalDepthISM(z,"disk",effWave)
        self.assertTrue(np.array_equal(result,opticalDepthISM))
        return

    def test_DustCharlotFallGetOpticalDepthClouds(self):
        z = 1.0
        effWave = 0.6563
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(z)
        # Check returns error for incorrect component
        with self.assertRaises(ValueError):
            self.DUST.getOpticalDepthClouds(z,"total",effWave)
        # Check calculation
        metalsName = "diskGasMetallicity"
        DATA = self.DUST.galaxies.get(z,properties=[metalsName])
        factorClouds = rcParams.getfloat("dustCharlotFall","opticalDepthCloudsFactor",fallback=1.0)
        wavelengthZeroPoint = rcParams.getfloat("dustCharlotFall","wavelengthZeroPoint",fallback=5500.0)
        wavelengthExponent = rcParams.getfloat("dustCharlotFall","wavelengthExponent",fallback=0.7)
        wavelengthRatio = (effWave/wavelengthZeroPoint)**wavelengthExponent
        localISMMetallicity = rcParams.getfloat("dustOpticalDepth","localISMMetallicity",fallback=0.02)
        opticalDepthClouds = factorClouds*(DATA[metalsName].data*metallicitySolar)/localISMMetallicity
        opticalDepthClouds /= wavelengthRatio
        result = self.DUST.getOpticalDepthClouds(z,"disk",effWave)
        self.assertTrue(np.array_equal(result,opticalDepthClouds))
        return

    def test_DustCharlotFallAttenuateStellarLuminosity(self):
        z = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(z)
        # Check raises error for incorrect names
        for name in ["totalLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000",
                     "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000:recent",
                     "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000",
                     "basicMass"]:
            with self.assertRaises(RuntimeError):
                DATA = self.DUST.attenuateStellarLuminosity(name,z)
        # Check calculation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000"
            MATCH = self.DUST.parseStellarLuminosityDatasetName(name)
            unattenuatedDatasetName = name.replace(":dustCharlotFall2000","")
            recentDatasetName = name.replace(":dustCharlotFall2000",":recent")
            PROPS = self.DUST.galaxies.get(z,properties=["redshift",unattenuatedDatasetName,
                                                         recentDatasetName])
            wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)/1.0e4
            opticalDepthISM = self.DUST.getOpticalDepthISM(z,MATCH.group('component'),wavelength)
            opticalDepthClouds = self.DUST.getOpticalDepthClouds(z,MATCH.group('component'),wavelength)
            attenuationISM = np.exp(-opticalDepthISM)
            attenuationClouds = np.exp(-opticalDepthClouds)
            print(attenuationClouds)
            print(attenuationISM)
            attenuatedLuminosity = ((PROPS[unattenuatedDatasetName].data-PROPS[recentDatasetName].data)
                                    + PROPS[recentDatasetName].data*attenuationClouds)*attenuationISM
            DATA = self.DUST.attenuateStellarLuminosity(name,z)            
            self.assertEqual(DATA.name,name)
            self.assertTrue(np.array_equal(DATA.data,attenuatedLuminosity))            
        return


    def test_DustCharlotFallAttenuateRecentStellarLuminosity(self):
        z = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(z)
        # Check raises error for incorrect names
        for name in ["totalLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000",
                     "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000",
                     "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000",
                     "basicMass"]:
            with self.assertRaises(RuntimeError):
                DATA = self.DUST.attenuateRecentStellarLuminosity(name,z)
        # Check calculation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000:recent"
            MATCH = self.DUST.parseStellarLuminosityDatasetName(name)
            recentDatasetName = name.replace(":dustCharlotFall2000","")
            PROPS = self.DUST.galaxies.get(z,properties=["redshift",recentDatasetName])
            wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)/1.0e4
            opticalDepthISM = self.DUST.getOpticalDepthISM(z,MATCH.group('component'),wavelength)
            opticalDepthClouds = self.DUST.getOpticalDepthClouds(z,MATCH.group('component'),wavelength)
            attenuationISM = np.exp(-opticalDepthISM)
            attenuationClouds = np.exp(-opticalDepthClouds)
            print(attenuationClouds)
            print(attenuationISM)
            attenuatedLuminosity = PROPS[recentDatasetName].data*attenuationClouds*attenuationISM
            DATA = self.DUST.attenuateRecentStellarLuminosity(name,z)            
            self.assertEqual(DATA.name,name)
            self.assertTrue(np.array_equal(DATA.data,attenuatedLuminosity))            
        return


    def test_DustCharlotFallAttenuateLineLuminosity(self):
        z = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(z)
        # Check raises error for incorrect names
        for name in ["totalLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000",
                     "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000",
                     "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000:recent",
                     "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000:recent",
                     "basicMass"]:
            with self.assertRaises(RuntimeError):
                DATA = self.DUST.attenuateLineLuminosity(name,z)
        # Check calculation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":dustCharlotFall2000"
            MATCH = self.DUST.parseLineLuminosityDatasetName(name)
            recentDatasetName = name.replace(":dustCharlotFall2000","")
            PROPS = self.DUST.galaxies.get(z,properties=["redshift",recentDatasetName])
            wavelength = getEffectiveWavelength(MATCH,PROPS["redshift"].data)/1.0e4
            opticalDepthISM = self.DUST.getOpticalDepthISM(z,MATCH.group('component'),wavelength)
            opticalDepthClouds = self.DUST.getOpticalDepthClouds(z,MATCH.group('component'),wavelength)
            attenuationISM = np.exp(-opticalDepthISM)
            attenuationClouds = np.exp(-opticalDepthClouds)
            print(attenuationClouds)
            print(attenuationISM)
            attenuatedLuminosity = PROPS[recentDatasetName].data*attenuationClouds*attenuationISM
            DATA = self.DUST.attenuateLineLuminosity(name,z)            
            self.assertEqual(DATA.name,name)
            self.assertTrue(np.array_equal(DATA.data,attenuatedLuminosity))            
        return

    def test_DustCharlotFallGet(self):
        z = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(z)
        for name in ["totalLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000","basicMass"]:
            with self.assertRaises(RuntimeError):
                DATA = self.DUST.get(name,1.0)
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        with warnings.catch_warnings():
            name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCharlotFall2000:recent"
            warnings.filterwarnings("ignore")
            DATA = self.DUST.get(name,redshift)
            self.assertEqual(DATA.name,name)
            self.assertIsNotNone(DATA.data)
        return
        

if __name__ == "__main__":
    unittest.main()




