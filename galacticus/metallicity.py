#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import unittest
from .datasets import Dataset
from .properties.manager import Property
from .constants import metallicitySolar


@Property.register_subclass('metallicity')
class Metallicity(Property):    
    """
    Metallicity: Compute galaxy metallicities.

    Functions: 
        matches(): Indicates whether specified dataset can be
                   processed by this class.  
        get(): Computes galaxy metallicty at specified redshift.
        parseDatasetName(): Parse the dataset name using regex.

    """
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def parseDatasetName(self,datasetName):
        """
        Metallicity.parseDatasetName(): Parse the specified dataset name using regex. Will
                                        return a re.search() instance if datasetName is
                                        '(disk|spheroid|total)Metallicity'. Otherwise will
                                        return a 'None' instance.
        
        USAGE:  SEARCH =  Metallicity.parseDatasetName(datasetName)

            INPUTS
                datasetName -- Dataset name to parse.

            OUTPUTS
                SEARCH      -- A re.search instance or None if datasetName is not a valid
                               metallicty dataset name.
              
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        searchString = "^(?P<component>disk|spheroid|total)Metallicity$"
        return re.search(searchString,datasetName)

    def matches(self,propertyName,redshift=None):
        """
        Metallicity.matches(): Returns boolean to indicate whether this class can process
                               the specified property.

        USAGE: match = Metallicty.matches(propertyName,[redshift=None])

           INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. (Redundant
                              in this particular case, but required for other properties.)

           OUTPUTS
              match        -- Boolean indicating whether this class can process
                              this property.

        """
        if self.parseDatasetName(propertyName):
            return True
        return False

    def get(self,propertyName,redshift):
        """
        Metallicity.get(): Compute galaxy metallicities for specified redshift.

        USAGE: DATA = Metallicity.get(propertyName,redshift)

           INPUTS
                propertyName -- Name of property to compute. This should be set 
                                to '(disk|spheroid|total)Metallicty'.  
                redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUT 
                DATA         -- Instance of galacticus.datasets.Dataset() class 
                                containing computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a metallicity."
            raise RuntimeError(msg)
        # Extract gas mass and galaxy radius
        gas = propertyName.replace("Metallicity","MassGas")
        metals = propertyName.replace("Metallicity","AbundancesGasMetals")
        GALS = self.galaxies.get(redshift,properties=[gas,metals])
        # Convert any zero values to NaN
        mass = np.copy(GALS[gas].data)
        np.place(mass,mass==0.0,np.nan)
        abundance = np.copy(GALS[metals].data)
        np.place(abundance,abundance==0.0,np.nan)
        # Clear GALS from memory
        del GALS
        # Compute metallicity
        DATA = Dataset(name=propertyName)
        DATA.data = np.copy(np.log10(abundance/mass))
        DATA.data -= np.log10(metallicitySolar)
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
        # Initialize the Metallicity class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.METAL = Metallicity(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.METAL.galaxies.GH5Obj.close()
        del self.METAL
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Metallicity: "+funcname)
        print("Testing Metallicity.matches() function")
        for component in ["disk","spheroid","total"]:
            self.assertTrue(self.METAL.matches(component+"Metallicity"))        
        self.assertFalse(self.METAL.matches("diskAbundanceMetals"))
        self.assertFalse(self.METAL.matches("totalAbundanceMetals"))
        print("TEST COMPLETE")
        print("\n")
        return

    def testGet(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Metallicity: "+funcname)
        print("Testing Metallicity.get() function")
        redshift = 1.0
        self.assertRaises(RuntimeError,self.METAL.get,"aMissingProperty",redshift)
        for component in ["disk","spheroid","total"]:
            DATA = self.METAL.get(component+"Metallicity",redshift)
            self.assertEqual(DATA.name,component+"Metallicity")
            OUT = self.METAL.galaxies.GH5Obj.selectOutput(redshift)
            properties = [component+"MassGas",component+"AbundancesGasMetals"]
            GALS = self.METAL.galaxies.get(redshift,properties=properties)
            mass = np.copy(GALS[component+"MassGas"].data)
            np.place(mass,mass==0.0,np.nan)
            abundance = np.copy(GALS[component+"AbundancesGasMetals"].data)
            np.place(abundance,abundance==0.0,np.nan)
            metallicity = np.copy(np.log10(abundance/mass)) - np.log10(metallicitySolar)
            for m,d in zip(metallicity,DATA.data):
                self.assertFalse(np.isinf(d))
                if np.isnan(m):
                    self.assertTrue(np.isnan(d))
                else:
                    diff = np.fabs(m-d)
                    self.assertLessEqual(diff,1.0e-6)
        print("TEST COMPLETE")
        print("\n")
        return
