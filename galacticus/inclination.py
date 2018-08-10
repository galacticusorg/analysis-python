#! /usr/bin/env python

import sys,os
import numpy as np
import unittest
from . import rcParams
from .datasets import Dataset
from .properties.manager import Property
from .constants import Pi

def Generate_Random_Inclinations(N,degrees=True):
    """ 
    Generate_Random_Inclinations: Return a list of N random inclination angles.

    USAGE: inc = Generate_Random_Inclinations(N,[degrees]) 
    
     INTPUTS
        N       : Integer number of angles to generate.                                                                                              
        degrees : Return angles in degrees? (Default value = True)                                                                                   
    
     OUTPUTS    
        inc     : Numpy array of N inclination angles.                                                                                               
    
    """
    angles = np.arccos(np.random.rand(N))
    if degrees:
        angles *= 180.0/Pi
    return angles

@Property.register_subclass('inclination')
class Inclination(Property):
    """
    Inclination: Compute galaxy inclinations.

    Functions:
            matches(): Indicates whether specified dataset can be processed by this class.
            get(): Computes galaxy inclinations at specified redshift.

    """    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None):
        """
        Inclination.matches(): Returns boolean to indicate whether this class can process
                               the specified property.

        USAGE: match =  Inclination.matches(propertyName,[redshift=None])

          INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. (Redundant
                              in this particular case, but required for other properties.)                              

          OUTPUTS
              match        -- Boolean indicating whether this class can process 
                              this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return propertyName == "inclination"

    def get(self,propertyName,redshift):        
        """
        Inclination.get(): Compute galaxy inclinations for specified redshift.
        
        USAGE:  DATA = Inclination.get(propertyName,redshift)
                
           INPUTS
           
                propertyName -- Name of property to compute. This should be set to 'inclination'.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.
           
           OUTPUT
                DATA         -- Instance of galacticus.datasets.Dataset() class containing 
                                computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            msg = funcname+"(): Specified property '"+propertyName+"' is not an inclination."
            raise RuntimeError(msg)
        degrees = rcParams.getboolean("inclination","degrees")
        N = self.galaxies.GH5Obj.countGalaxies(redshift)
        inclination = Generate_Random_Inclinations(N,degrees=degrees)
        inclination = np.random.random(N)
        DATA = Dataset(name="inclination",data=inclination,attr={"degrees":degrees})
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
        # Initialize the Inclination class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.INC = Inclination(GALS)
        return
    
    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.INC.galaxies.GH5Obj.close()
        del self.INC
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def testGet(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Inclination: "+funcname)
        print("Testing Inclination.get() function")
        redshift = 1.0
        DATA = self.INC.get("inclination",redshift)
        N = self.INC.galaxies.GH5Obj.countGalaxies(redshift)
        self.assertIsNotNone(DATA)
        self.assertEqual(DATA.name,"inclination")
        self.assertIsNotNone(DATA.data)
        self.assertEqual(DATA.data.size,N)
        self.assertTrue(len(DATA.attr.keys())>0)
        self.assertTrue("degrees" in DATA.attr.keys())
        self.assertRaises(RuntimeError,self.INC.get,"notAnInclination",redshift)
        print("TEST COMPLETE")
        print("\n")
        return

    def testMatches(self):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Inclination: "+funcname)
        print("Testing Inclination.matches() function")
        self.assertTrue(self.INC.matches("inclination"))
        self.assertFalse(self.INC.matches("redshift"))
        print("TEST COMPLETE")
        print("\n")
        return

    
