#! /usr/bin/env python

import sys
import re
import numpy as np
import unittest
from .. import rcParams
from ..datasets import Dataset
from ..properties.manager import Property

@Property.register_subclass('dustParameters')
class DustParameters(Property):
    """
    DustParameters: Compute the dust parameters, A_V and R_V.

    Functions:
            parseDatasetName(): Parse a dust parameter dataset.
            matches(): Indicates whether specified dataset can be processed by this class.
            getAttenuationParameter(): Compute A_X given attenuated and unattenuated 
                                       X-band luminosities.
            getReddeningParameter(): Compute R_V given attenuated and unattenuated 
                                       V-band and B-band luminosities.
            get(): Computes A_V and R_V at specified redshift.

    """    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def parseDatasetName(self,propertyName):
        """
        DustParameters.parseDatasetName: Parse a dust parameters dataset.

        USAGE: SEARCH = DustParameters.parseDatasetName(propertyName)

             INPUTS
                propertyName -- Property name to parse.

             OUTPUTS

                SEARCH       -- Regex search (re.search) object or None if
                                propertyName cannot be parsed.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        searchString = "^(?P<component>[^:]+)LuminositiesStellar"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            ":dust(?P<label>[^:]+):(?P<parameter>A|R)_V"
        MATCH = re.search(searchString,propertyName)
        return MATCH

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        DustParameters.matches(): Returns boolean to indicate whether this class can process
                                 the specified property.

        USAGE: match =  DustParameters.matches(propertyName,[redshift=None],[raiseError=False])

          INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.                           
              raiseError   -- Raise error if property does not match.
                              (Default = False)  

          OUTPUTS
              match        -- Boolean indicating whether this class can process 
                              this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid dust parameter dataset. "
            raise RuntimeError(msg)
        return False
    
    def getAttenuationParameter(self,attenL,unattenL):                
        """
        DustParameters.getAteenuationParameter(): Compute attenuation parameter.
        
        USAGE:  A = DustParameters.getAttenuationParameter(attenL,unattenL)
                
           INPUTS           
                attenL   -- Attenuated luminosity.
                unattenL -- Un-attenuated luminosity.
           
           OUTPUT
                A        -- Numpy array of attenuation dust parameters.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        nonZero     = unattenL > 0.0        
        A           = np.ones_like(unattenL)*np.nan
        A[nonZero] = -2.5*np.log10(attenL[nonZero]/unattenL[nonZero])
        return A

    def getReddeningParameter(self,attenV,unattenV,attenB,unattenB):
        """
        DustParameters.getReddening(): Compute reddening parameter.
        
        USAGE:  R = DustParameters.getReddeningParameter(attenV,unattenV,
                                                         attenB,unattenB)
                
           INPUTS           
                attenV   -- Attenuated V-band luminosity.
                unattenV -- Un-attenuated V-band luminosity.
                attenB   -- Attenuated B-band luminosity.
                unattenB -- Un-attenuated B-band luminosity.
           
           OUTPUT
                R        -- Numpy array of reddening parameters.

        """
        AV = self.getAttenuationParameter(attenV,unattenV)
        AB = self.getAttenuationParameter(attenB,unattenB)
        colorExcess = AB - AV
        RV = AV/colorExcess
        return RV
                
    def get(self,propertyName,redshift):        
        """
        DustParameters.get(): Compute dust model parameters for specified redshift.
        
        USAGE:  DATA = DustParameters.get(propertyName,redshift)
                
           INPUTS
           
                propertyName -- Name of property to compute.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.
           
           OUTPUT
                DATA         -- Instance of galacticus.datasets.Dataset() class containing 
                                computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Extract information from property name
        propertyMatch = self.parseDatasetName(propertyName)
        component      = propertyMatch.group("component"     )
        redshiftLabel  = propertyMatch.group("redshiftString")
        label          = propertyMatch.group("label"         )
        parameter      = propertyMatch.group("parameter"     )
        # Build names of the attenuated and unattenuated luminosity datasets.
        unattenuatedVDatasetName = component+"LuminositiesStellar:V:rest"+redshiftLabel
        attenuatedVDatasetName   = unattenuatedVDatasetName+":dust"+label
        unattenuatedBDatasetName = component+"LuminositiesStellar:B:rest"+redshiftLabel
        attenuatedBDatasetName   = unattenuatedBDatasetName+":dust"+label
        # Retrieve the luminosities.
        propertyNames = [ attenuatedVDatasetName, unattenuatedVDatasetName ]
        if (parameter == "R"):
            propertyNames.extend([ attenuatedBDatasetName, unattenuatedBDatasetName ])
        PROPS = self.galaxies.get(redshift,properties=propertyNames)
        # Compute the required parameter.
        DATA = Dataset(name=propertyName)
        if (parameter == "A"):
            DATA.data = np.copy(self.getAttenuationParameter(PROPS[attenuatedVDatasetName].data,\
                                                         PROPS[unattenuatedVDatasetName].data))
        elif (parameter == "R"):
            DATA.data = np.copy(self.getReddeningParameter(PROPS[attenuatedVDatasetName].data,\
                                                       PROPS[unattenuatedVDatasetName].data,\
                                                       PROPS[attenuatedBDatasetName].data,\
                                                       PROPS[unattenuatedBDatasetName].data))
        else:
            raise ValueError(funcname+"(): Parameter '"+parameter\
                                 +"'not recognized. Should be A or R.")
        del PROPS
        return DATA
    

class UnitTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        from ..galaxies import Galaxies
        from ..io import GalacticusHDF5
        from ..data import GalacticusData
        from shutil import copyfile
        # Locate the dynamic version of the galacticus.snapshotExample.hdf5 file.
        DATA = GalacticusData(verbose=False)
        self.snapshotFile = DATA.searchDynamic("galacticus.snapshotExample.hdf5")
        self.removeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.snapshotFile is None:
            self.snapshotFile = DATA.dynamic+"/examples/galacticus.snapshotExample.hdf5"
            self.removeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.snapshotExample.hdf5",self.snapshotFile)
        # Initialize the DustParameters class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.DUST = DustParameters(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DUST.galaxies.GH5Obj.close()
        del self.DUST
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def testParseDatasetName(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Parameters: "+funcname)
        print("Testing DustParameters.parseDatasetName() function")
        names = ["diskLuminositiesStellar:z1.000:dustCompendium:A_V",\
                     "spheroidLuminositiesStellar:z1.000:dustCalzetti:R_V",\
                     "totalLuminositiesStellar:z1.000:dustAllen_AV1.0:R_V"]
        for name in names:
            self.assertIsNotNone(self.DUST.parseDatasetName(name))
        names = ["totalLuminositiesStellar:z1.000:dustAllen_AV1.0:X_V",\
                     "totalLuminositiesStellar:z1.000:A_V",\
                     "diskLuminositiesStellar:z1.000:dustCompendium"]
        for name in names:
            self.assertIsNone(self.DUST.parseDatasetName(name))
        print("TEST COMPLETE")
        print("\n")
        return

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Parameters: "+funcname)
        print("Testing DustParameters.matches() function")
        names = ["diskLuminositiesStellar:z1.000:dustCompendium:A_V",\
                     "spheroidLuminositiesStellar:z1.000:dustCalzetti:R_V",\
                     "totalLuminositiesStellar:z1.000:dustAllen_AV1.0:R_V"]
        for name in names:      
            self.assertTrue(self.DUST.matches(name))
        names = ["totalLuminositiesStellar:z1.000:dustAllen_AV1.0:X_V",\
                     "totalLuminositiesStellar:z1.000:A_V",\
                     "diskLuminositiesStellar:z1.000:dustCompendium"]
        for name in names:
            self.assertFalse(self.DUST.matches(name,raiseError=False))
            self.assertRaises(RuntimeError,self.DUST.matches,name,raiseError=True)
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetAttenuationParameter(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Parameters: "+funcname)
        print("Testing DustParameters.getAttenuationParameter() function")
        N = 50
        unattenL = np.ones(N,dtype=float)
        attenL = np.maximum(np.random.rand(N)*unattenL,1.0e-10)
        value = -2.5*np.log10(attenL/unattenL)
        result = self.DUST.getAttenuationParameter(attenL,unattenL)
        self.assertTrue(type(result),np.ndarray)
        diff = np.fabs(result-value)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        mask = np.random.rand(N)<0.1
        np.place(unattenL,mask,0.0)
        result = self.DUST.getAttenuationParameter(attenL,unattenL)
        [self.assertTrue(np.isnan(result[i])) for i in range(len(result)) if mask[i]]
        print("TEST COMPLETE")
        print("\n")
        return            

    def testGetReddeningParameter(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Dust Parameters: "+funcname)
        print("Testing DustParameters.getReddeningParameter() function")
        N = 50
        unattenV = np.ones(N,dtype=float)
        attenV = np.maximum(np.random.rand(N)*unattenV,1.0e-10)
        unattenB = np.ones(N,dtype=float)
        attenB = np.maximum(np.random.rand(N)*unattenB,1.0e-10)
        AV = -2.5*np.log10(attenV/unattenV)
        AB = -2.5*np.log10(attenB/unattenB)
        value = AV/(AB-AV)
        result = self.DUST.getReddeningParameter(attenV,unattenV,attenB,unattenB)
        self.assertTrue(type(result),np.ndarray)
        diff = np.fabs(result-value)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        mask = np.random.rand(N)<0.1
        np.place(unattenV,mask,0.0)
        mask2 = np.random.rand(N)<0.1
        np.place(unattenB,mask2,0.0)
        mask = np.logical_or(mask,mask2)
        result = self.DUST.getReddeningParameter(attenV,unattenV,attenB,unattenB)
        [self.assertTrue(np.isnan(result[i])) for i in range(len(result)) if mask[i]]
        print("TEST COMPLETE")
        print("\n")
        return            
        
