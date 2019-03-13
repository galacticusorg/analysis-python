#! /usr/bin/env python

import sys,os,re
import unittest
import numpy as np
import warnings
from shutil import copyfile
import galacticus.properties.manager
from galacticus.bulgeToTotal import BulgeToTotal
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData


class TestBulgeToTotal(unittest.TestCase):
    
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
        # Initialize the Totals class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.BULGE = BulgeToTotal(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.BULGE.galaxies.GH5Obj.close()
        del self.BULGE
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_BulgeToTotalMatches(self):
        redshift = 1.0
        for name in ["MassStellar","StarFormationRate"]:
            self.assertTrue(self.BULGE.matches("bulgeToTotal"+name,redshift))
            self.assertFalse(self.BULGE.matches("diskMassStellar",redshift))
            with self.assertRaises(RuntimeError):
                self.BULGE.matches("diskMassStellar",redshift,raiseError=True)
        return

    def test_BulgeToTotalParseDatasetName(self):
        for name in ["MassStellar","StarFormationRate"]:
            self.assertIsNotNone(self.BULGE.parseDatasetName("bulgeToTotal"+name))            
            self.assertIsNone(self.BULGE.parseDatasetName("total"+name))
        return

    def test_BulgeToTotalGet(self):
        redshift = 1.0
        with self.assertRaises(RuntimeError):
            self.BULGE.get("aMissingProperty",redshift)
        for name in ["MassStellar","StarFormationRate"]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                OUT = self.BULGE.galaxies.GH5Obj.selectOutput(redshift)
                bulge = np.array(OUT["nodeData/spheroid"+name])
                disk = np.array(OUT["nodeData/disk"+name])
                data = bulge/(bulge+disk)
                DATA = self.BULGE.get("bulgeToTotal"+name,redshift)
                self.assertIsNotNone(DATA)
                self.assertEqual(DATA.name,"bulgeToTotal"+name)
                self.assertIsNotNone(DATA.data)
                diff = np.fabs(data-DATA.data)
                for i in range(len(diff)):
                    if np.isnan(diff[i]):
                        self.assertEqual(disk[i]+bulge[i],0.0)
                    else:
                        self.assertLessEqual(diff[i],1.0e-6)
        return

if __name__ == "__main__":
    unittest.main()
        
        
