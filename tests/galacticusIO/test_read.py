#! /usr/bin/env python

import sys,os
import numpy as np
import unittest
from shutil import copyfile
from galacticus.properties.manager import Property
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.io.read import ReadHDF5
from galacticus.data import GalacticusData



class TestRead(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Locate the dynamic version of thegalacticus.snapshotExample.hdf5 file.
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
        # Initialize the ReadHDF5 class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.READ = ReadHDF5(GALS)
        return

        
    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.READ.galaxies.GH5Obj.close()
        del self.READ
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_ReadMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        redshift = 1.0
        property = self.READ.galaxies.GH5Obj.availableDatasets(redshift)[0]
        self.assertTrue(self.READ.matches(property,redshift))
        self.assertFalse(self.READ.matches("aMissingProperty",redshift))
        return

    def test_ReadGet(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        redshift = 1.0
        self.assertRaises(RuntimeError,self.READ.get,"aMissingProperty",redshift)
        property = self.READ.galaxies.GH5Obj.availableDatasets(redshift)[0]
        OUT = self.READ.galaxies.GH5Obj.selectOutput(redshift)
        data = np.array(OUT["nodeData/"+property])
        DATA = self.READ.get(property,redshift)
        self.assertIsNotNone(DATA)
        self.assertEqual(DATA.name,property)
        self.assertIsNotNone(DATA.data)
        diff = np.fabs(data-DATA.data)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        return

if __name__ == "__main__":
    unittest.main()
