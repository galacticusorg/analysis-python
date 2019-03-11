#! /usr/bin/env python

import sys,os,re
import numpy as np
import unittest
import copy
from .. import rcParams
from ..datasets import Dataset
from ..properties.manager import Property
from ..constants import Pi,megaParsec,milli,centi
from ..constants import massAtomic,massSolar,massFractionHydrogen
from .CompendiumTable import CompendiumTable

@Property.register_subclass('dustOpticalDepthCentral')
class DustOpticalDepthCentral(Property):
    """
    DustOpticalDepthCentral(): Compute dust optical depths through the centers of galaxy disks.

    Methods:
           parseDatasetName(): Parse optical depth dataset names
           matches(): Indicates whether specified dataset can be processed by this class.
           computeColumnDensityMetals(): Compute column density of metals in galaxy disks.
           getOpacity(): Return the opacity either from compendium file or by approximation.
           get(): Compute dust optical depths through centers of galaxy disks at specified reshift.

    """
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def parseDatasetName(self,propertyName):
        """
        DustOpticalDepthCentral.parseDatasetName(): Parse a dust optical depth dataset.

        USAGE:  SEARCH = DustOpticalDepthCentral.parseDatasetName(propertyName)

           INPUTS
                propertyName -- Property name to parse.

           OUTPUTS
                SEARCH       -- Regex seearch (re.search) object or None if
                                propertyName cannot be parsed.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        searchString = "^diskDustOpticalDepthCentral:dust(?P<dust>Atlas|Compendium)$"
        MATCH = re.search(searchString,propertyName)
        if MATCH is not None:
            return MATCH
        return None

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        DustOpticalDepthCentral.matches(): Returns boolean to indicate whether this
                                        class can process the specified property.
        
        USAGE: matches = DustOpticalDepthCentral.matches(propertyName,[redshift=None],
                                                         [raiseError=False])

           INPUTS 
               propertyName -- Name of property to process.
                   redshift -- Redshift value to query Galacticus HDF5 outputs.  
                               (Redundant in this particular case, but required 
                               for other properties.)  
                 raiseError -- Raise error if property does not match. (Default = False)

          OUTPUTS
                match       -- Boolean indicating whether this class can process
                               this property.   

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+"' is not valid."+\
                " Use 'diskDustOpticalDepthCentral:dust(Atlas|Compendium)'."
            raise RuntimeError(msg)
        return False
    
    def computeColumnDensityMetals(self,redshift):
        """
        DustOpticalDepthCentral.computeColumnDensityMetals(): Compute the column density of metals in the
                                                              disk of the galaxy.

        USAGE: density = DustOpticalDepthCentral.computeColumnDensityMetals(redshift)

            INPUT
               redshift -- Redshift value to query Galacticus HDF5 outputs.  
            OUTPUT
               density  -- Numpy array of column densities.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        PROPS = self.galaxies.get(redshift,properties=["diskAbundancesGasMetals","diskRadius"])
        columnDensityMetals = np.ones_like(PROPS['diskAbundancesGasMetals'].data)*np.nan
        mask = np.logical_and(PROPS['diskRadius'].data>0.0,PROPS['diskAbundancesGasMetals'].data>=0.0)
        columnDensityMetals[mask] = np.copy(PROPS['diskAbundancesGasMetals'].data[mask])
        columnDensityMetals[mask] /= (2.0*Pi*np.copy(PROPS['diskRadius'].data[mask])**2)
        return columnDensityMetals

    def getOpacity(self,dustLabel):        
        """
        DustOpticalDepthCentral.getOpacity(): Return the opacity of through the center of the galaxy.

        USAGE:  opacity = DustOpticalDepthCentral.getOpacity(dustLabel)

            INPUT  
                dustLabel -- String to indicate dust method to use ('Atlas' or 'Compendium')

            OUTPUTS
                opacity   -- Numpy array storing opacities of galaxies.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if dustLabel == "Compendium":
            # Get opacity in cm^2/g
            COMPENDIUM = CompendiumTable()
            COMPENDIUM.loadOpacity()
            opacity = copy.copy(COMPENDIUM.opacity)
            del COMPENDIUM
        elif dustLabel == "Atlas":
            # Approximate opacity
            # i) specify necessary parameters
            localISMMetallicity = 0.02  # ... Metallicity in the local ISM.
            AV_to_EBV = 3.10            # ... (A_V/E(B-V); Savage & Mathis 1979)
            NH_to_EBV = 5.8e21          # ... (N_H/E(B-V); atoms/cm^2/mag; Savage & Mathis 1979)                                                                               
            opticalDepthToMagnitudes = 2.5*np.log10(np.exp(1.0))
            # ii) compute opacity in cm^2/g
            opacity = (AV_to_EBV/opticalDepthToMagnitudes)/NH_to_EBV
            opacity *= (massFractionHydrogen/(massAtomic/milli))/localISMMetallicity
        else:
            raise ValueError(funcname+"(): Dust label '"+dustLabel+"' not recognized. "+\
                                 "Should be 'Atlas' or 'Compendium'.")
        return opacity

    def get(self,propertyName,redshift):
        """
        DustOpticalDepthCentral.get(): Return the dust optical depth through the center of the
                                       galactic disk.
        
        USAGE:  DATA = DustOpticalDepthCentral.get(propertyName,redshift)

            INPUTS
               propertyName -- Name of property to compute. This should be set to 
                               'diskDustOpticalDepthCentral:dust(Atlas|Compendium)'.
               redshift     -- Redshift value to query Galacticus HDF5 outputs.                                                                                                
        
            OUTPUT
               DATA         -- Instance of galacticus.datasets.Dataset() class                                                                                                 
                               containing computed galaxy information.    
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        # Get column density for metals
        columnDensityMetals = self.computeColumnDensityMetals(redshift)               
        # Get the dust-to-metals ratio. If not provided we use a
        # default of 0.44 which is approximately correct for the Milky
        # Way (e.g. Popping et al.; 2017;
        # http://adsabs.harvard.edu/abs/2017MNRAS.471.3152P).
        dustToMetalsRatio = rcParams.getfloat("dustOpticalDepth","dustToMetalsRatio",fallback=0.44)
        # Find column density of dust through center of disks [Msolar/Mpc^2].
        columnDensityDust = columnDensityMetals*dustToMetalsRatio        
        # Correct column density units to g/cm^2
        columnDensityDust *= (massSolar/milli)*(centi/megaParsec)**2
        # Get opacity in cm^2/g
        opacity = self.getOpacity(MATCH.group("dust")) 
        # Compute optical depth
        DATA = Dataset(name=propertyName)
        DATA.data = np.copy(columnDensityDust*opacity)
        return DATA


class UnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Intitialize class  
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
        # Initialize the DustOpticalDepthCentral class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.DEPTH = DustOpticalDepthCentral(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DEPTH.galaxies.GH5Obj.close()
        del self.DEPTH
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def testParseDatasetName(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: DustOpticalDepthCentral: "+funcname)
        print("Testing DustOpticalDepthCentral.parseDatasetName() function")        
        for dust in ["Atlas","Compendium"]:
            name = "diskDustOpticalDepthCentral:dust"+dust
            MATCH = self.DEPTH.parseDatasetName(name)
            self.assertIsNotNone(MATCH)
            self.assertEqual(MATCH.group("dust"),dust)
        for name in ["spheroidDustOpticalDepthCentral:dustAtlas",
                     "diskDustOpticalDepthCentral:dustAltas",
                     "diskDustOpticalDepthCentral:dustAtlasClouds",
                     "diskDustOpticalDepthCentral:dust"
                     ]:
            MATCH = self.DEPTH.parseDatasetName(name)
            self.assertIsNone(MATCH)
        print("TEST COMPLETE")
        print("\n")
        return

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: DustOpticalDepthCentral: "+funcname)
        print("Testing DustOpticalDepthCentral.matches() function")
        for dust in ["Atlas","Compendium"]:
            name = "diskDustOpticalDepthCentral:dust"+dust
            self.assertTrue(self.DEPTH.matches(name))
        for name in ["spheroidDustOpticalDepthCentral:dustAtlas",
                     "diskDustOpticalDepthCentral:dustAltas",
                     "diskDustOpticalDepthCentral:dustAtlasClouds",
                     "diskDustOpticalDepthCentral:dust"
                     ]:            
            self.assertFalse(self.DEPTH.matches(name))
            self.assertRaises(RuntimeError,self.DEPTH.matches,name,raiseError=True)
        print("TEST COMPLETE")
        print("\n")
        return
    
    def testComputeColumnDensityMetals(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: DustOpticalDepthCentral: "+funcname)
        print("Testing  DustOpticalDepthCentral.computeColumnDensityMetals() function")
        density = self.DEPTH.computeColumnDensityMetals(1.0)
        self.assertEqual(type(density),np.ndarray)
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetOpacity(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: DustOpticalDepthCentral: "+funcname)
        print("Testing  DustOpticalDepthCentral.getOpacity() function")
        for dust in ["Atlas","Compendium"]:
            opacity = self.DEPTH.getOpacity(dust)
            self.assertEqual(type(opacity),np.float64)
        for name in ["atlas","compendium","Altas","dustAtlas","AtlasClouds"]:
            self.assertRaises(ValueError,self.DEPTH.getOpacity,name)
        print("TEST COMPLETE")
        print("\n")
        return        

    def testGet(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: DustOpticalDepthCentral: "+funcname)
        print("Testing  DustOpticalDepthCentral.getOpacity() function")
        for dust in ["Atlas","Compendium"]:
            name = "diskDustOpticalDepthCentral:dust"+dust
            DATA = self.DEPTH.get(name,1.0)
            self.assertEqual(DATA.name,name)
            self.assertEqual(type(DATA.data),np.ndarray)
        for name in ["spheroidDustOpticalDepthCentral:dustAtlas",
                     "diskDustOpticalDepthCentral:dustAltas",
                     "diskDustOpticalDepthCentral:dustAtlasClouds",
                     "diskDustOpticalDepthCentral:dust"
                     ]:            
            self.assertRaises(RuntimeError,self.DEPTH.get,name,1.0)
        print("TEST COMPLETE")
        print("\n")
        return

