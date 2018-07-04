#! /usr/bin/env python

import __future__
import numpy as np
import sys,os,fnmatch,re
from ..datasets import Dataset
from ..errors import ParseError
from ..constants import massSolar

def parseStellarMass(datasetName):
    """
    parseStellarMass(): Parse a stellar mass dataset name using Regex.

    USAGE: MATCH = parseStellarMass(name)

       INPUTS
          name -- Stellar mass dataset name (e.g. diskMassStellar, or totalMassStellar).
      OUTPUTS
         MATCH -- Regex search instance.

    """
    funcname = sys._getframe().f_code.co_name
    searchString = "(?P<component>\w+)MassStellar"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH

def parseStarFormationRate(datasetName):
    """
    parseStarFormationRate(): Parse a star formation rate dataset name using Regex.

    USAGE: MATCH = parseStarFormationRate(name)

       INPUTS
          name -- Star formation rate dataset name (e.g. diskStarFormationRate, or 
                  totalStarFormationRate).
      OUTPUTS
         MATCH -- Regex search instance.

    """
    funcname = sys._getframe().f_code.co_name
    searchString = "(?P<component>\w+)StarFormationRate"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH



class GalacticusStellarMass(object):
    """
    GalacticusStellarMass(): Class to read/compute galaxy stellar mass from Galacticus HDF5 file.

    Functions:
    
      getStellarMass: Return Dataset class instance containng stellar mass.
      getDataset: Return Dataset class instance containing stellar mass.

    """
    def __init__(self,GH5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.GH5Obj = GH5Obj
        self.verbose = verbose
        return

    def __call__(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getStellarMass(datasetName,z)
    
    def getStellarMass(self,datasetName,z):
        """
        GalacticusStellarMass.getStellarMass(): Return a Dataset class instance containing the galaxy
                                                stellar masses at specified redshift, z.

        USAGE: DATA = GalacticusStellarMass.getStellarMass(name,z)

          INPUTS
              name -- Name of stellar mass dataset to extract.
              z    -- Redshift of output at which to extract stellar mass.
         
         OUTPUTS
              DATA -- Dataset class instance containing stellar mass information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = parseStellarMass(datasetName)
        if MATCH.group("component")=="total":            
            diskName = datasetName.replace("total","disk")
            DISK = self.getStellarMass(diskName,z)
            sphereName = datasetName.replace("total","spheroid")
            SPHERE = self.getStellarMass(sphereName,z)
            DATA = Dataset(name=datasetName,path=DISK.path,unitsInSI=DISK.unitsInSI)
            DATA.data = np.copy(DISK.data+SPHERE.data)
            del DISK,SPHERE
        else:
            DATA = self.GH5Obj.getDataset(datasetName,z)
        return DATA
                    
    def getDataset(self,datasetName,z):
        """
        GalacticusStellarMass.getDataset(): Return a Dataset class instance containing the galaxy
                                            stellar masses at specified redshift, z.

        USAGE: DATA = GalacticusStellarMass.getDataset(name,z)

          INPUTS
              name -- Name of stellar mass dataset to extract.
              z    -- Redshift of output at which to extract stellar mass.
         
         OUTPUTS
              DATA -- Dataset class instance containing stellar mass information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getStellarMass(datasetName,z)


class GalacticusStarFormationRate(object):
    """
    GalacticusStarFormationRate(): Class to read/compute galaxy star formation rate from Galacticus HDF5 file.

    Functions:
    
      getStarFormationRate: Return Dataset class instance containng star formation rate.
      getDataset: Return Dataset class instance containing star formation rate.

    """
    def __init__(self,GH5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.GH5Obj = GH5Obj
        self.verbose = verbose
        return
    
    def __call__(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getStarFormationRate(datasetName,z)

    def getStarFormationRate(self,datasetName,z):
        """
        GalacticusStarFormationRate.getStarFormationRate(): Return a Dataset class instance containing the galaxy
                                                            star formation rate  at specified redshift, z.

        USAGE: DATA = GalacticusStarFormationRate.getStarFormationRate(name,z)

          INPUTS
              name -- Name of star formation rate dataset to extract.
              z    -- Redshift of output at which to extract star formation rate.
         
         OUTPUTS
              DATA -- Dataset class instance containing star formation rate information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = parseStarFormationRate(datasetName)
        if MATCH.group("component")=="total":
            diskName = datasetName.replace("total","disk")
            DISK = self.getStarFormationRate(diskName,z)
            sphereName = datasetName.replace("total","spheroid")
            SPHERE = self.getStarFormationRate(sphereName,z)
            DATA = Dataset(name=datasetName,path=DISK.path,unitsInSI=DISK.unitsInSI)
            DATA.data = np.copy(DISK.data+SPHERE.data)
            del DISK,SPHERE
        else:
            DATA = self.GH5Obj.getDataset(datasetName,z)
        return DATA
                    
    def getDataset(self,datasetName,z):
        """
        GalacticusStarFormationRate.getStarDataset(): Return a Dataset class instance containing the galaxy
                                                      star formation rate  at specified redshift, z.

        USAGE: DATA = GalacticusStarFormationRate.getDataset(name,z)

          INPUTS
              name -- Name of star formation rate dataset to extract.
              z    -- Redshift of output at which to extract star formation rate.
         
         OUTPUTS
              DATA -- Dataset class instance containing star formation rate information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getStarFormationRate(datasetName,z)
    
    

        
