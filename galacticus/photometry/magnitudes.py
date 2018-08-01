#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from ..datasets import Dataset
from ..constants import luminosityAB
from ..errors import ParseError
from .luminosities import GalacticusStellarLuminosity

def parseMagnitude(datasetName):
    """
    parseMagnitude(): Parse a magnitude dataset name using Regex.

    USAGE: MATCH = parseMagnitude(name)
    
    INPUTS 
        name -- Magnitude dataset name.  

    OUTPUTS 
       MATCH -- Regex search instance.

    """
    funcname = sys._getframe().f_code.co_name
    searchString = "^(?P<component>disk|spheroid|total)Magnitude(?P<magnitude>Apparent|Absolute):"+\
        "(?P<filter>[^:]+):(?P<frame>[^:]+)(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
        "(?P<recent>:recent)?(?P<dust>:dust[^:]+)?(?P<system>:vega|:AB)?"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


class GalacticusMagnitude(GalacticusStellarLuminosity):
    """
    GalacticusMagnitude(): Class to read/compute galaxy stellar magnitudes from 
                           Galacticus HDF5 file.                                                                                                                   

    Base class: GalacticusStellarLuminosity

    Functions:
      getDataset: Return Dataset class instance containing stellar magnitude.
      getDistanceModulus: Return the distance module for given cosmology.
      getMagnitudes: Return Dataset class instance containing stellar magnitude.      
      getOffsetVegaAB: Return AB-Vega offset for given filter.

    """

    def __init__(self,GH5Obj,path=None,vbandFilterName="Buser_V",\
                     spectrumFile="A0V_Castelli.xml",verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(GalacticusMagnitude,self).__init__(GH5Obj,verbose=verbose)
        self.FILTERS = GalacticusFilter(path=path,vbandFilterName=vbandFilterName,\
                                            spectrumFile=spectrumFile,verbose=verbose)        
        return

    def __call__(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return self.getMagnitude(datasetName,z)
    

    def getOffsetVegaAB(self,filterName,keepInMemory=True,kRomberg=8,**kwargs):
        """
        GalacticusMagnitude.getOffsetVegaAB(): Return AB-Vega offset for specified filter.
        
        USAGE:  offset = GalacticusMagnitude.getOffsetVegaAB(filterName,[keepInMemory=True],\
                                                             [kRomberg=8],[**kwargs])
                                                             
        INPUTS
            filterName   -- Name of the filter. 
            keepInMemory -- Store filter in memory for future use. [Default=True] 
            kRomberg     -- Number of k-nodes for Romberg integration. [Default=8] 
            **kwrgs      -- Keywords arguments to pass to scipy.interpolate.interp1d.
        
        OUTPUTS
            offset       -- AB-Vega offset for specified filter.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        FILTER = self.FILTERS.load(filterName,keepInMemory=keepInMemory,kRomberg=kRomberg,**kwargs)
        return FILTER.vegaOffset
    
    def getDistanceModulus(self,z):
        """
        GalacticusMagnitude.getDistanceModulus(): Return distance modulus according to redshift of
                                                  galaxy.
        
        USAGE:  mod =  GalacticusMagnitude.getDistanceModulus(z)        

           INPUTS
               z  -- Redshift to locate nearest snapshot.
               
          OUTPUTS
              mod -- Distance modulus at redshift of galaxy.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if "lightconeRedshift" in self.GH5.availableDatasets(z):
            OUT = self.GH5.selectOutput(z)
            redshift = np.array(OUT["nodeData/lightconeRedshift"])
        else:
            n = self.GH5.countGalaxiesAtRedshift(z)
            redshift = np.ones(n,dtype=float)*self.GH5.nearestRedshift(z)
        # TO DO -- extract distance modulus from cosmology module.        
        MOD = 0.0
        return MOD


    def getMagnitude(self,datasetName,z,keepInMemory=True,kRomberg=8,**kwargs):
        """
        GalacticusMagnitude.getMagnitude(): Return a Dataset class instance containing the galaxy
                                            magnitude information.
                                            
        USAGE:  DATA = GalacticusMagnitude.getMagnitude(datasetName,z,[keepInMemory=True],\
                                                        [kRomberg=8],[**kwargs])

          INPUTS
              datasetName  -- Name of the magnitude dataset to extract.
              z            -- Redshift of output fro which to extract magnitude.
              keepInMemory -- Store filter in memory for future use. [Default=True] 
              kRomberg     -- Number of k-nodes for Romberg integration. [Default=8] 
              **kwrgs      -- Keywords arguments to pass to scipy.interpolate.interp1d.
              
          OUTPUTS
              DATA         -- Dataset class instance containing magnitude information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName in self.GH5Obj.availableDatasets(z):
            return GH5Obj.getDataset(datasetName,z)
        MATCH = parseMagnitude(datasetName)
        # Extract luminosity
        luminosityName = MATCH.group('component')+"LuminositiesStellar:"+MATCH.group('filterName')+\
            ":"+MATCH.group('frame')+MATCH.group('redshiftString')
        if MATCH.group('recent') is not None:
            luminosityName = luminosityName + MATCH.group('recent')
        if MATCH.group('dust') is not None:
            luminosityName = luminosityName + MATCH.group('dust')
        DATA = self.getStellarLuminosity(luminosityName,z)
        DATA.name = datasetName
        DATA.unitsInSI = None
        # Compute absolute magnitude
        DATA.data = -2.5*np.log10(DATA.data+1.0e-40)
        # Convert to Vega magnitudes if necessary
        if MATCH.group('system') is not None:
            if fnmatch.fnmatch(MATCH.group('system').lower(),"vega"):
                DATA.data += self.getOffsetVegaAB(MATCH.group('filterName'))
        # Convert to apparent magnitude if required
        if fnmatch.fnmatch(MATCH.group('magnitude').lower(),"app*"):            
            DATA.data += self.getDistanceModulus(z)
        return DATA


    def getDataset(self,datasetName,z,keepInMemory=True,kRomberg=8,**kwargs):
        """
        GalacticusMagnitude.getDataset(): Return a Dataset class instance containing the galaxy
                                          magnitude information.
                                            
        USAGE:  DATA = GalacticusMagnitude.getDataset(datasetName,z,[keepInMemory=True],\
                                                      [kRomberg=8],[**kwargs])

          INPUTS
              datasetName  -- Name of the magnitude dataset to extract.
              z            -- Redshift of output fro which to extract magnitude.
              keepInMemory -- Store filter in memory for future use. [Default=True] 
              kRomberg     -- Number of k-nodes for Romberg integration. [Default=8] 
              **kwrgs      -- Keywords arguments to pass to scipy.interpolate.interp1d.
              
          OUTPUTS
              DATA         -- Dataset class instance containing magnitude information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        DATA = self.getMagnitude(datasetName,z,keepInMemory=keepInMemory,\
                                     kRomberg=kRomberg,**kwargs)
        return DATA
