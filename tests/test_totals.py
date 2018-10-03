#! /usr/bin/env python

import unittest
import numpy as np
from shutil import copyfile
import galacticus.properties.manager
from galacticus.totals import Totals
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData

class TestTotals(unittest.TestCase):

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
        self.TOTALS = Totals(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.TOTALS.galaxies.GH5Obj.close()
        del self.TOTALS
        if self.removeExample:
            os.remove(self.snapshotFile)
        return
    
    def test_TotalsMatches(self):
        # Testing totals.matches
        redshift = 1.0
        self.assertTrue(self.TOTALS.matches("totalMassStellar",redshift))
        for name in ["diskMassStellar","spheroidMassStellar","totalMetallicity",
                     "totalMagnitudeAbsolute:SDSS_r:z1.000"]:            
            self.assertFalse(self.TOTALS.matches(name,redshift))
            with self.assertRaises(RuntimeError):
                self.TOTALS.matches(name,redshift,raiseError=True)
        return

    def test_TotalsGet(self):
        # Testing totals.get
        redshift = 1.0
        with self.assertRaises(RuntimeError):
            self.TOTALS.get("aMissingProperty",redshift)
        for name in ["MassStellar","StarFormationRate"]:
            OUT = self.TOTALS.galaxies.GH5Obj.selectOutput(redshift)
            data = np.array(OUT["nodeData/spheroid"+name]) + np.array(OUT["nodeData/disk"+name])
            DATA = self.TOTALS.get("total"+name,redshift)
            self.assertIsNotNone(DATA)
            self.assertEqual(DATA.name,"total"+name)
            self.assertIsNotNone(DATA.data)
            diff = np.fabs(data-DATA.data)
            [self.assertLessEqual(d,1.0e-6) for d in diff]
        return


if __name__ == "__main__":
    unittest.main()
