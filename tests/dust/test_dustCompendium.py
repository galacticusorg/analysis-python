#! /usr/bin/env python

import numpy as np
import warnings
import unittest
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.dust.dustCompendium import DustCompendium


class TestDustCompendium(unittest.TestCase):

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
        self.DUST = DustCompendium(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DUST.galaxies.GH5Obj.close()
        del self.DUST
        if self.removeExample:
            os.remove(self.snapshotFile)
        return    

    def test_DustCompendiumParseDatasetName(self):        
        for component in ["disk","spheroid"]:
            for frame in ["rest","observed"]:
                name = component+"LuminositiesStellar:SDSS_r:"+frame+":z1.000:dustCompendium"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = name + ":recent"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = component+"LineLuminosity:balmerAlpha6563:"+frame+":z1.000:dustCompendium"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = name + ":recent"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = component+"LineLuminosity:balmerAlpha6563:"+frame+":SDSS_r:z1.000:dustCompendium"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
                name = name + ":recent"
                self.assertIsNotNone(self.DUST.parseDatasetName(name))
        name = "totalLuminositiesStellar:SDSS_r:rest:z1.000:dustCompendium"
        self.assertIsNone(self.DUST.parseDatasetName(name))
        name = "basicMass"
        self.assertIsNone(self.DUST.parseDatasetName(name))
        return

    def test_DustCompendiumMatches(self):
        for component in ["disk","spheroid"]:
            for frame in ["rest","observed"]:
                name = component+"LuminositiesStellar:SDSS_r:"+frame+":z1.000:dustCompendium"
                self.assertTrue(self.DUST.matches(name))
                name = name + ":recent"
                self.assertTrue(self.DUST.matches(name))
                name = component+"LineLuminosity:balmerAlpha6563:"+frame+":z1.000:dustCompendium"
                self.assertTrue(self.DUST.matches(name))
                name = name + ":recent"
                self.assertTrue(self.DUST.matches(name))
                name = component+"LineLuminosity:balmerAlpha6563:"+frame+":SDSS_r:z1.000:dustCompendium"
                self.assertTrue(self.DUST.matches(name))
                name = name + ":recent"
                self.assertTrue(self.DUST.matches(name))
        for name in ["totalLuminositiesStellar:SDSS_r:rest:z1.000:dustCompendium","basicMass"]:            
            self.assertFalse(self.DUST.matches(name))
            with self.assertRaises(RuntimeError):
                self.DUST.matches(name,raiseError=True)
        return

    def test_DustCompendiumGet(self):
        for name in ["totalLuminositiesStellar:SDSS_r:rest:z1.000:dustCompendium","basicMass"]:
            with self.assertRaises(RuntimeError):
                DATA = self.DUST.get(name,1.0)
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        with warnings.catch_warnings():
            name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCompendium"
            warnings.filterwarnings("ignore")
            DATA = self.DUST.get(name,redshift)
            self.assertEqual(DATA.name,name)
            self.assertIsNotNone(DATA.data)
        return
        

if __name__ == "__main__":
    unittest.main()




