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
from galacticus.nodes import HostNode


class TestHostNode(unittest.TestCase):

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
        self.HOST = HostNode(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.HOST.galaxies.GH5Obj.close()
        del self.HOST
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_HostNodeGetHostIndex(self):    
        isCentral = np.array([0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,0,1,1,0,1])
        index = np.array([1,1,5,5,5,5,6,9,9,9,11,11,13,13,15,15,19,19,19,19,20,22,22])
        self.assertTrue(np.array_equal(index,self.HOST.getHostIndex(isCentral)))
        return

    def test_HostNodeMatches(self):
        z = 1.0
        for name in self.HOST.galaxies.GH5Obj.availableDatasets(z):
            self.assertTrue(self.HOST.matches(name+":host"))
            self.assertFalse(self.HOST.matches(name,raiseError=False))
            with self.assertRaises(RuntimeError):
                self.HOST.matches(name,raiseError=True)
        return
                                    
    def test_HostNodeGet(self):
        z = 1.0
        name = "nodeMass200.0"
        hostName = name+":host"
        GALS = self.HOST.galaxies.get(z,properties=[name,"nodeIsIsolated"])        
        DATA = self.HOST.get(hostName,z)
        index = self.HOST.getHostIndex(GALS["nodeIsIsolated"].data)
        truth = GALS[name].data[index]
        self.assertEqual(DATA.name,hostName)
        self.assertTrue(np.allclose(DATA.data,truth))
        self.assertEqual(DATA.attr["unitsInSI"],GALS[name].attr["unitsInSI"])
        return
    
if __name__ == "__main__":
    unittest.main()

