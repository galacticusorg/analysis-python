#! /usr/bin/env python

import sys,os
import numpy as np
import unittest
import warnings
from .datasets import Dataset
from .properties.manager import Property


@Property.register_subclass('bulgetototal')
class BulgeToTotal(Property):
    """
    BulgeToTotal: Compute a bulge-to-total ratio based upon specified galaxy property.

    Functions: 
            matches(): Indicates whether specified dataset can be processed by this class.  
            get(): Computes bulge-to-total ratio at specified redshift.

    """    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None):
        """
        BulgeToTotal.matches(): Returns boolean to indicate whether this class can 
                                process the specified property.

        USAGE: match =  BulgeToTotal.matches(propertyName,[redshift=None])

          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. 

          OUTPUTS 
              match        -- Boolean indicating whether this class can process 
                              this property.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return propertyName.startswith("bulgeToTotal")
    
    def get(self,propertyName,redshift):
        """
        BulgeToTotal.get(): Compute bulge-to-total ratio for specified galaxy property 
                            using total and spheroid components at specified redshift.

        USAGE: DATA = BulgeToTotal.get(propertyName,redshift)

           INPUTS
             propertyName -- Name of bulge-to-total ratio to compute. This name
                             shoud start with 'bulgeToTotal'.
             redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUTS
              DATA        -- Instance of galacticus.datasets.Dataset() class 
                             containing computed galaxy information, or None
                             if one of the components is missing.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            raise RuntimeError(funcname+"(): Cannot process property '"+propertyName+"'.")
        # Get spheroid and total properties
        spheroid = propertyName.replace("bulgeToTotal","spheroid")
        total = propertyName.replace("bulgeToTotal","total")
        GALS = self.galaxies.get(redshift,properties=[spheroid,total])
        if any([GALS[key] is None for key in GALS.keys()]):
            return None
        # Compute ratio and return result
        DATA = Dataset(name=propertyName)
        DATA.attr = {}
        DATA.data = np.copy(GALS[spheroid].data/GALS[total].data)
        del GALS
        return DATA

class UnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from .galaxies import Galaxies
        from .io import GalacticusHDF5
        from .data import GalacticusData
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

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: BulgeToTotal: "+funcname)
        print("Testing BulgeToTotal.matches() function")
        redshift = 1.0
        self.assertTrue(self.BULGE.matches("bulgeToTotalMassStellar",redshift))
        self.assertFalse(self.BULGE.matches("diskMassStellar",redshift))
        self.assertFalse(self.BULGE.matches("spheroidMassStellar",redshift))
        print("TEST COMPLETE")
        print("\n")
        return

    def testGet(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: BulgeToTotal: "+funcname)
        print("Testing BulgeToTotal.get() function")
        redshift = 1.0
        self.assertRaises(RuntimeError,self.BULGE.get,"aMissingProperty",redshift)
        for name in ["MassStellar","StarFormationRate"]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",message="invalid value encountered in divide")
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
        print("TEST COMPLETE")
        print("\n")
        return
