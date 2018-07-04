#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from ..datasets import Dataset
from ..constants import luminosityAB
from ..errors import ParseError

def parseStellarLuminosity(datasetName):
    """
    parseStellarLuminosity(): Parse a stellar luminosity dataset name using Regex.

    USAGE: MATCH = parseStellarLuminosity(name)

       INPUTS 
           name -- Stellar luminosity dataset name.
      OUTPUTS 
          MATCH -- Regex search instance.

    """
    funcname = sys._getframe().f_code.co_name
    searchString = "^(?P<component>disk|spheroid|total)LuminositiesStellar:(?P<filter>[^:]+):(?P<frame>[^:]+)"+\
        "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH

def parseBulgeToTotal(datasetName):
    """
    parseBulgeToTotal(): Parse a stellar luminosity bulge/total ratio dataset name using Regex.

    USAGE: MATCH = parseBulgeToTotal(name)

       INPUTS 
           name -- Stellar luminosity bulge/total ratio dataset name.
      OUTPUTS 
          MATCH -- Regex search instance.

    """
    funcname = sys._getframe().f_code.co_name
    searchString = "bulgeToTotalLuminosities:(?P<filter>[^:]+):(?P<frame>[^:]+)"+\
        "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH



class GalacticusStellarLuminosity(object):
    """
    GalacticusStellarLuminosity(): Class to read/compute galaxy stellar luminosities 
                                   from Galacticus HDF5 file.
    
    Functions:
      availableLuminosities: Return list of stellar luminosities currently available in 
                             the HDF5 file.
      getStellarLuminosity: Return Dataset class instance containing stellar luminosity.
      getDataset: Return Dataset class instance containing stellar luminosity.    

    """
    def __init__(self,GH5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.GH5Obj = GH5Obj
        self.verbose = verbose
        return

    def __call__(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getStellarLuminosity(datasetName,z)
    
    def availableLuminosities(self,z):
        """
        GalacticusStellarLuminosity.availableLuminosities(): Return list of stellar luminosities that
                                                             currently available in the HDF5 in the 
                                                             output closest to the specified redshift.

        USAGE: luminosities = GalacticusStellarLuminosity.availableLuminosities(z)

           INPUTS
              z            -- Redshift to query.

          OUTPUTS
              luminosities -- List of stellar luminosities available at nearest output.
                                                             
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        properties = self.GH5Obj.availableDatasets(z)
        return fnmatch.filter(properties,"*LuminositiesStellar:*")
    
    def getStellarLuminosity(self,datasetName,z):
        """ 
        GalacticusStellarLuminosity.getStellarLuminosity(): Return a Dataset class instance containing
                                                            the galaxy stellar luminosity at specified 
                                                            redshift, z.

        USAGE: DATA = GalacticusStellarLuminosity.getStellarLuminosity(name,z)

          INPUTS 
                name -- Name of stellar luminosity dataset to extract.  
                   z -- Redshift of output at which to extract stellar luminosity

        OUTPUTS 
               DATA -- Dataset class instance containing stellar luminosity information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName in self.GH5Obj.availableDatasets(z):
            return self.GH5Obj.getDataset(datasetName,z)
        MATCH = parseStellarLuminosity(datasetName)
        if MATCH.group("component")=="total":            
            diskName = datasetName.replace("total","disk")
            DISK = self.getStellarLuminosity(diskName,z)
            sphereName = datasetName.replace("total","spheroid")
            SPHERE = self.getStellarLuminosity(sphereName,z)
            DATA = Dataset(name=datasetName,path=DISK.path,unitsInSI=DISK.unitsInSI)
            DATA.data = np.copy(DISK.data+SPHERE.data)
            del DISK,SPHERE
        else:
            if MATCH.group('dust') is not None:
                # Extract dust
                noDustName = datasetName.replace(MATCH.group('dust'),"")
                DATA = self.GH5Obj.getDataset(noDustName,z)
                DATA.name = datasetName
                # TO DO -- add in dust options
                pass
            else:
                DATA = self.GH5Obj.getDataset(datasetName,z)
        return DATA


    def getDataset(self,datasetName,z):
        """
        GalacticusStellarLuminosity.getDataset(): Return a Dataset class instance containing
                                                  the galaxy stellar luminosity at specified 
                                                  redshift, z.

        USAGE: DATA = GalacticusStellarLuminosity.getDataset(name,z)

          INPUTS 
                name -- Name of stellar luminosity dataset to extract.  
                   z -- Redshift of output at which to extract stellar luminosity

        OUTPUTS 
               DATA -- Dataset class instance containing stellar luminosity information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getStellarLuminosity(datasetName,z)



class GalacticusBulgeToTotal(object):
    """
    GalacticusBulgeToTotal(): Class to read/compute galaxy stellar luminosity
                              bulge-to-total ratios from Galacticus HDF5 file.
    
    Functions:
      getBulgeToTotal: Return Dataset class instance containing stellar bulge-to-total ratio..
      getDataset: Return Dataset class instance containing stellar bulge-to-total ratio.    

    """
    def __init__(self,GH5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.GH5Obj = GH5Obj
        self.verbose = verbose
        return

    def __call__(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getBulgeToTotal(datasetName,z)


    def getBulgeToTotal(self,datasetName,z):
        """ 
        GalacticusBulgeToTotal.getBulgeTotal(): Return a Dataset class instance containing
                                                the galaxy bulge-to-total ratio at specified 
                                                redshift, z.

        USAGE: DATA = GalacticusBulgeToTotal.getBulgeToTotal(name,z)

          INPUTS 
                name -- Name of bulge-to-total ratio dataset to extract.  
                   z -- Redshift of output at which to extract bulge-to-total ratio.

        OUTPUTS 
               DATA -- Dataset class instance containing bulge-to-total ratio information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName in self.GH5Obj.availableDatasets(z):
            return self.GH5Obj.getDataset(datasetName,z)
        MATCH = parseBulgeToTotal(datasetName)
        LUM = GalacticusStellarLuminosity()
        bulgeName = datasetName.replace("bulgeToTotalLuminosities","spheroidLuminositiesStellar")
        BULGE = LUM.getStellarLuminosity(bulgeName,z)
        totalName = datasetName.replace("bulgeToTotalLuminosities","totalLuminositiesStellar")
        TOTAL = LUM.getStellarLuminosity(totalName,z)
        DATA = Dataset(name=datasetName,path=TOTAL.path,unitsInSI=1.0)
        DATA.data = np.copy(BULGE.data/TOTAL.data)
        del BULGE,TOTAL
        return DATA

    def getDataset(self,datasetName,z):
        """ 
        GalacticusBulgeToTotal.getDataset(): Return a Dataset class instance containing
                                             the galaxy bulge-to-total ratio at specified 
                                             redshift, z.

        USAGE: DATA = GalacticusBulgeToTotal.getDataset(name,z)

          INPUTS 
                name -- Name of bulge-to-total ratio dataset to extract.  
                   z -- Redshift of output at which to extract bulge-to-total ratio.

        OUTPUTS 
               DATA -- Dataset class instance containing bulge-to-total ratio information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getBulgeTotal(datasetName,z)
