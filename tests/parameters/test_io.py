#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import warnings
import unittest
from galacticus.data import GalacticusData
from galacticus.io import GalacticusHDF5
from galacticus.parameters import GalacticusParameters
from galacticus.parameters.io import ParametersFromHDF5
from galacticus.strings import removeByteStrings

class TestReadFromHDF5(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # Locate the dynamic version of the galacticus.snapshotExample.hdf5 file.
        DATA = GalacticusData()
        snapshotFile = DATA.search("galacticus.snapshotExample.hdf5")        
        paramfile = DATA.search("snapshotExample.xml")
        self.GH5 = GalacticusHDF5(snapshotFile,'r')
        self.PARAMS = GalacticusParameters(file=paramfile)
        self.PARAMS.mapTree()
        return

    @classmethod
    def tearDownClass(self):
        self.GH5.close()
        return

    def test_readFromGalacticusFile(self):        
        PARAMS = ParametersFromHDF5.read(self.GH5)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for path in PARAMS.map:
                result = PARAMS.getParameter(path)
                truth = self.PARAMS.getParameter(path)
            if truth is not None:
                self.assertEqual(removeByteStrings(result),truth)
        return


if __name__ == "__main__":
    unittest.main()


    
