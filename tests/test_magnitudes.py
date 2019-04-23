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
from galacticus.magnitudes import Magnitude

class TestMagnitude(unittest.TestCase):

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
        # Initialize the Magnitude class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.MAGS = Magnitude(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.MAGS.galaxies.GH5Obj.close()
        del self.MAGS
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_MagnitudeParseDatasetName(self):        
        for comp in ["disk","spheroid","total"]:
            for mag in ["Absolute","Apparent"]:
                for frame in ["rest","observed"]:
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000"
                    self.assertIsNotNone(self.MAGS.parseDatasetName(name))
                    MATCH = self.MAGS.parseDatasetName(name)
                    self.assertEqual(MATCH.group("component"),comp)
                    self.assertEqual(MATCH.group("magnitude"),mag)
                    self.assertEqual(MATCH.group("frame"),frame)
                    self.assertEqual(MATCH.group("filter"),"SDSS_r")
                    self.assertEqual(MATCH.group("redshift"),"1.000")
                    self.assertIsNone(MATCH.group("system"))
                    self.assertIsNone(MATCH.group("recent"))
                    self.assertIsNone(MATCH.group("dust"))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega"
                    self.assertIsNotNone(self.MAGS.parseDatasetName(name))
                    MATCH = self.MAGS.parseDatasetName(name)
                    self.assertEqual(MATCH.group("component"),comp)
                    self.assertEqual(MATCH.group("magnitude"),mag)
                    self.assertEqual(MATCH.group("frame"),frame)
                    self.assertEqual(MATCH.group("filter"),"SDSS_r")
                    self.assertEqual(MATCH.group("redshift"),"1.000")
                    self.assertEqual(MATCH.group("system"),":vega")
                    self.assertIsNone(MATCH.group("recent"))
                    self.assertIsNone(MATCH.group("dust"))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:AB"
                    self.assertIsNotNone(self.MAGS.parseDatasetName(name))
                    MATCH = self.MAGS.parseDatasetName(name)
                    self.assertEqual(MATCH.group("component"),comp)
                    self.assertEqual(MATCH.group("magnitude"),mag)
                    self.assertEqual(MATCH.group("frame"),frame)
                    self.assertEqual(MATCH.group("filter"),"SDSS_r")
                    self.assertEqual(MATCH.group("redshift"),"1.000")
                    self.assertEqual(MATCH.group("system"),":AB")
                    self.assertIsNone(MATCH.group("recent"))
                    self.assertIsNone(MATCH.group("dust"))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega:recent"
                    self.assertIsNotNone(self.MAGS.parseDatasetName(name))
                    MATCH = self.MAGS.parseDatasetName(name)
                    self.assertEqual(MATCH.group("component"),comp)
                    self.assertEqual(MATCH.group("magnitude"),mag)
                    self.assertEqual(MATCH.group("frame"),frame)
                    self.assertEqual(MATCH.group("filter"),"SDSS_r")
                    self.assertEqual(MATCH.group("redshift"),"1.000")
                    self.assertEqual(MATCH.group("system"),":vega")
                    self.assertEqual(MATCH.group("recent"),":recent")
                    self.assertIsNone(MATCH.group("dust"))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega:recent:dustCompendium"
                    self.assertIsNotNone(self.MAGS.parseDatasetName(name))
                    MATCH = self.MAGS.parseDatasetName(name)
                    self.assertEqual(MATCH.group("component"),comp)
                    self.assertEqual(MATCH.group("magnitude"),mag)
                    self.assertEqual(MATCH.group("frame"),frame)
                    self.assertEqual(MATCH.group("filter"),"SDSS_r")
                    self.assertEqual(MATCH.group("redshift"),"1.000")
                    self.assertEqual(MATCH.group("system"),":vega")
                    self.assertEqual(MATCH.group("recent"),":recent")
                    self.assertEqual(MATCH.group("dust"),":dustCompendium")
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega:dustCalzetti_Av0.1"
                    self.assertIsNotNone(self.MAGS.parseDatasetName(name))
                    MATCH = self.MAGS.parseDatasetName(name)
                    self.assertEqual(MATCH.group("component"),comp)
                    self.assertEqual(MATCH.group("magnitude"),mag)
                    self.assertEqual(MATCH.group("frame"),frame)
                    self.assertEqual(MATCH.group("filter"),"SDSS_r")
                    self.assertEqual(MATCH.group("redshift"),"1.000")
                    self.assertEqual(MATCH.group("system"),":vega")
                    self.assertEqual(MATCH.group("dust"),":dustCalzetti_Av0.1")
                    self.assertIsNone(MATCH.group("recent"))
        names = ["basicMass","diskMagnitude:SDSS_r:rest:z1.000",
                 "diskApparentMagnitude:SDSS_r:rest:z1.000",
                 "diskAbsoluteMagnitude:SDSS_r:rest:z1.000",
                 "diskMagnitudeapparent:SDSS_r:rest:z1.000",
                 ]
        for name in names:
            self.assertIsNone(self.MAGS.parseDatasetName(name))
        return

    def test_MagnitudeMatches(self):        
        for comp in ["disk","spheroid","total"]:
            for mag in ["Absolute","Apparent"]:
                for frame in ["rest","observed"]:
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:AB"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:recent"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega:recent"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:AB:recent"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:dustCompendium"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega:dustCompendium"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:AB:dustCompendium"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:dustCalzetti_Av0.1"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega:dustCalzetti_Av0.1"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:AB:dustCalzetti_Av0.1"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:recent:dustCharlotFall2000"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:vega:recent:dustCharlotFall2000"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:AB:recent:dustCharlotFall2000"
                    self.assertTrue(self.MAGS.parseDatasetName(name))
        names = ["basicMass","diskMagnitude:SDSS_r:rest:z1.000",
                 "diskApparentMagnitude:SDSS_r:rest:z1.000",
                 "diskAbsoluteMagnitude:SDSS_r:rest:z1.000",
                 "diskMagnitudeapparent:SDSS_r:rest:z1.000",
                 ]
        for name in names:
            self.assertFalse(self.MAGS.matches(name,raiseError=False))
            with self.assertRaises(RuntimeError):
                self.MAGS.matches(name,raiseError=True)
        return

    def test_MagnitudeGetLuminosityName(self):
        for comp in ["disk","spheroid","total"]:
            for mag in ["Absolute","Apparent"]:
                for frame in ["rest","observed"]:
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000"
                    lumName = comp+"LuminositiesStellar:SDSS_r:"+frame+":z1.000"
                    self.assertEqual(lumName,self.MAGS.getLuminosityName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:recent"
                    lumName = comp+"LuminositiesStellar:SDSS_r:"+frame+":z1.000:recent"                    
                    self.assertEqual(lumName,self.MAGS.getLuminosityName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:recent:dustCompendium"
                    lumName = comp+"LuminositiesStellar:SDSS_r:"+frame+":z1.000:recent:dustCompendium"                    
                    self.assertEqual(lumName,self.MAGS.getLuminosityName(name))
                    name = comp+"Magnitude"+mag+":SDSS_r:"+frame+":z1.000:dustCompendium"
                    lumName = comp+"LuminositiesStellar:SDSS_r:"+frame+":z1.000:dustCompendium"                    
                    self.assertEqual(lumName,self.MAGS.getLuminosityName(name))
        return
        
    def test_MagnitudeGetVegaOffset(self):
        name = "diskMagnitudeAbsolute:SDSS_r:rest:z1.0000"
        self.assertEqual(0.0,self.MAGS.getVegaOffset(name))
        name = "diskMagnitudeAbsolute:SDSS_r:rest:z1.0000:AB"
        self.assertEqual(0.0,self.MAGS.getVegaOffset(name))
        name = "diskMagnitudeAbsolute:SDSS_r:rest:z1.0000:vega"
        truth = -0.139302055718797
        offset = self.MAGS.getVegaOffset(name)        
        diff = np.fabs(offset-truth)/np.fabs(truth)
        self.assertLessEqual(diff,1.0e-4)
        return
    
    def test_MagnitudeGet(self):
        z = 1.0
        zStr = self.MAGS.galaxies.GH5Obj.getRedshiftString(z)
        # Test absolute magnitudes
        name = "diskMagnitudeAbsolute:SDSS_r:rest:"+zStr
        lumName = "diskLuminositiesStellar:SDSS_r:rest:"+zStr
        GALS = self.MAGS.galaxies.get(z,properties=[lumName])
        zeroCorrection = rcParams.getfloat("magnitude","zeroCorrection",fallback=1.0e-50)
        truth = -2.5*np.log10(GALS[lumName].data+zeroCorrection)
        DATA = self.MAGS.get(name,z)
        self.assertEqual(DATA.name,name)
        self.assertTrue(np.allclose(truth,DATA.data))
        name = "totalMagnitudeAbsolute:SDSS_r:rest:"+zStr+":AB"
        lumName = "totalLuminositiesStellar:SDSS_r:rest:"+zStr
        GALS = self.MAGS.galaxies.get(z,properties=[lumName])
        zeroCorrection = rcParams.getfloat("magnitude","zeroCorrection",fallback=1.0e-50)
        truth = -2.5*np.log10(GALS[lumName].data+zeroCorrection)
        DATA = self.MAGS.get(name,z)
        self.assertEqual(DATA.name,name)
        self.assertTrue(np.allclose(truth,DATA.data))
        name = "totalMagnitudeAbsolute:SDSS_r:observed:"+zStr+":vega"
        lumName = "totalLuminositiesStellar:SDSS_r:observed:"+zStr
        GALS = self.MAGS.galaxies.get(z,properties=[lumName])
        zeroCorrection = rcParams.getfloat("magnitude","zeroCorrection",fallback=1.0e-50)
        truth = -2.5*np.log10(GALS[lumName].data+zeroCorrection) + -0.139302055718797
        DATA = self.MAGS.get(name,z)
        self.assertEqual(DATA.name,name)
        self.assertTrue(np.allclose(truth,DATA.data))
        # Test apparent magnitudes
        name = "diskMagnitudeApparent:SDSS_r:rest:"+zStr
        lumName = "diskLuminositiesStellar:SDSS_r:rest:"+zStr
        GALS = self.MAGS.galaxies.get(z,properties=[lumName,"redshift"])
        zeroCorrection = rcParams.getfloat("magnitude","zeroCorrection",fallback=1.0e-50)
        truth = -2.5*np.log10(GALS[lumName].data+zeroCorrection)
        dm = self.MAGS.galaxies.GH5Obj.cosmology.band_corrected_distance_modulus(GALS["redshift"].data)
        truth += dm
        DATA = self.MAGS.get(name,z)
        self.assertEqual(DATA.name,name)
        self.assertTrue(np.allclose(truth,DATA.data))
        name = "totalMagnitudeApparent:SDSS_r:rest:"+zStr+":AB"
        lumName = "totalLuminositiesStellar:SDSS_r:rest:"+zStr
        GALS = self.MAGS.galaxies.get(z,properties=[lumName,"redshift"])
        zeroCorrection = rcParams.getfloat("magnitude","zeroCorrection",fallback=1.0e-50)
        truth = -2.5*np.log10(GALS[lumName].data+zeroCorrection)
        dm = self.MAGS.galaxies.GH5Obj.cosmology.band_corrected_distance_modulus(GALS["redshift"].data)
        truth += dm
        DATA = self.MAGS.get(name,z)
        self.assertEqual(DATA.name,name)
        self.assertTrue(np.allclose(truth,DATA.data))
        name = "totalMagnitudeApparent:SDSS_r:observed:"+zStr+":vega"
        lumName = "totalLuminositiesStellar:SDSS_r:observed:"+zStr
        GALS = self.MAGS.galaxies.get(z,properties=[lumName,"redshift"])
        zeroCorrection = rcParams.getfloat("magnitude","zeroCorrection",fallback=1.0e-50)
        truth = -2.5*np.log10(GALS[lumName].data+zeroCorrection) + -0.139302055718797
        dm = self.MAGS.galaxies.GH5Obj.cosmology.band_corrected_distance_modulus(GALS["redshift"].data)
        truth += dm
        DATA = self.MAGS.get(name,z)
        self.assertEqual(DATA.name,name)
        self.assertTrue(np.allclose(truth,DATA.data))
        # Check returns None for missing luminosities
        name = "diskMagnitudeApparent:SDSS_X:rest:"+zStr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.assertIsNone(self.MAGS.get(name,z))
        # Test crashes for bad names
        names = ["basicMass","diskMagnitude:SDSS_r:rest:z1.000",
                 "diskApparentMagnitude:SDSS_r:rest:z1.000",
                 "diskAbsoluteMagnitude:SDSS_r:rest:z1.000",
                 "diskMagnitudeapparent:SDSS_r:rest:z1.000"
                 ]
        for name in names:
            with self.assertRaises(RuntimeError):
                self.MAGS.get(name,z)        
        return


if __name__ == "__main__":
    unittest.main()
