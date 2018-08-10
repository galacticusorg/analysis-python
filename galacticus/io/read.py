#! /usr/bin/env python

import sys,os
import numpy as np
import unittest
from ..properties.manager import Property

@Property.register_subclass('readhdf5')
class ReadHDF5(Property):
    """
    Read: Manage reading of galaxy data from HDF5 file.

    Functions: 
          matches(): Indicates whether specified dataset can be
                     processed by this class.  
          get(): Extracts galaxy property at specified redshift.

    """
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None):
        """
        Read.matches(): Returns boolean to indicate whether this class can 
                        process the specified property.

        USAGE: match = Read.matches(propertyName,[redshift=None])

          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.

          OUTPUTS 
              match        -- Boolean indicating whether this class can 
                              process this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.galaxies.GH5Obj.datasetExists(propertyName,redshift)

    def get(self,propertyName,redshift=None):
        """
        Read.get(): Extract galaxy property for specified redshift.

        USAGE:  DATA = Inclination.get(propertyName,redshift)

          INPUTS 
             propertyName -- Name of property to extract. 
             redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUT 
             DATA         -- Instance of galacticus.datasets.Dataset()
                             class containing computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName,redshift=redshift):
            msg = funcname+"(): Cannot locate '"+propertyName+"' in Galacticus HDF5 file."
            raise RuntimeError(msg)
        return self.galaxies.GH5Obj.getDataset(propertyName,redshift)


class UnitTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        from ..galaxies import Galaxies
        from ..io import GalacticusHDF5
        from ..data import GalacticusData
        from shutil import copyfile
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
        # Initialize the Inclination class.
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

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: ReadHDF5: "+funcname)
        print("Testing ReadHDF5.matches() function")
        redshift = 1.0
        property = self.READ.galaxies.GH5Obj.availableDatasets(redshift)[0]
        self.assertTrue(self.READ.matches(property,redshift))
        self.assertFalse(self.READ.matches("aMissingProperty",redshift))
        print("TEST COMPLETE")
        print("\n")
        return

    def testGet(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: ReadHDF5: "+funcname)
        print("Testing ReadHDF5.get() function")
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
        print("TEST COMPLETE")
        print("\n")
        return
    
