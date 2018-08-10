#! /usr/bin/env python

import sys,os
import numpy as np
import unittest
from .datasets import Dataset
from .properties.manager import Property

@Property.register_subclass('totals')
class Totals(Property):
    """
    Totals: Compute a total property by summing up the disk and spheroid components.

    Functions: 
            matches(): Indicates whether specified dataset can be processed by this class.  
            get(): Computes galaxy total at specified redshift.

    """    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return
        
    def matches(self,propertyName,redshift=None):
        """
        Totals.matches(): Returns boolean to indicate whether this class can 
                          process the specified property.

        USAGE: match =  Totals.matches(propertyName,[redshift=None])                                                                                                       
        
          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. 

          OUTPUTS 
              match        -- Boolean indicating whether this class can process 
                              this property.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return propertyName.startswith("total")
    
    def get(self,propertyName,redshift):
        """
        Totals.get(): Compute total galaxy property using disk and spheroid 
                      components at specified redshift.

        USAGE: DATA = Totals.get(propertyName,redshift)

           INPUTS
             propertyName -- Name of total property to compute. This name
                             shoud start with 'total'.
             redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUTS
              DATA        -- Instance of galacticus.datasets.Dataset() class 
                             containing computed galaxy information, or None
                             if one of the components is missing.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            raise RuntimeError(funcname+"(): Cannot process property '"+propertyName+"'.")
        # Get disk and spheroid properties
        components = [propertyName.replace("total","disk"),propertyName.replace("total","spheroid")]
        GALS = self.galaxies.get(redshift,properties=components)
        if any([GALS[key] is None for key in GALS.keys()]):
            return None
        # Sum components and return total
        DATA = Dataset(name=propertyName)
        DATA.attr = GALS[components[0]].attr
        DATA.data = np.copy(GALS[components[0]].data+GALS[components[1]].data)
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

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Totals: "+funcname)
        print("Testing Totals.matches() function")
        redshift = 1.0
        self.assertTrue(self.TOTALS.matches("totalMassStellar",redshift))
        self.assertFalse(self.TOTALS.matches("diskMassStellar",redshift))
        self.assertFalse(self.TOTALS.matches("spheroidMassStellar",redshift))        
        print("TEST COMPLETE")
        print("\n")
        return

    def testGet(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Totals: "+funcname)
        print("Testing Totals.get() function")
        redshift = 1.0
        self.assertRaises(RuntimeError,self.TOTALS.get,"aMissingProperty",redshift)
        for name in ["MassStellar","StarFormationRate"]:
            OUT = self.TOTALS.galaxies.GH5Obj.selectOutput(redshift)
            data = np.array(OUT["nodeData/spheroid"+name]) + np.array(OUT["nodeData/disk"+name])
            DATA = self.TOTALS.get("total"+name,redshift)
            self.assertIsNotNone(DATA)
            self.assertEqual(DATA.name,"total"+name)
            self.assertIsNotNone(DATA.data)
            diff = np.fabs(data-DATA.data)
            [self.assertLessEqual(d,1.0e-6) for d in diff]
        print("TEST COMPLETE")
        print("\n")
        return


