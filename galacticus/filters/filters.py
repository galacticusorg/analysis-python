#! /usr/bin/env python

import sys,fnmatch
import numpy as np
import unittest
from . import Filter
from .vega import Vega
from .topHats import TopHat
from ..data import GalacticusData


class GalacticusFilter(object):    
    """
    GalacticusFilter: Class to manage reading and writing of filter transmissions used 
                      with Galacticus.

    USAGE: GalFilter = GalacticusFilter([path=None],[vbandFilterName=Buser_V],\
                                        [spectrumFile=A0V_Castelli.xml],[verbose=False])

        INTPUT 
           verbose -- Increase verbosity. [Default=False]

    Note: Upon initialization the class will create an instance of the Vega class. 
          This are necessary for computing AB/Vega filter offsets.

    Functions:

             clearCache:  Clear all isntances of filters stored in memory.
             create: Create an instance of the Filter class with specified information.
             write: Write Filter class instance to an XML file.
             load: Load a filter from an XML file.

    """    
    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(self.__class__,self).__init__()
        self.verbose = verbose
        self.DATA = GalacticusData(verbose=self.verbose)
        self.VEGA = Vega(verbose=self.verbose)
        self.TOPHATS = TopHat(verbose=self.verbose,VegaObj=self.VEGA)
        self.cache = {}
        return

    def clearCache(self):
        """
        GalacticusFilter.clearCache: Clear filter objects stored in memory.
          
        USAGE   GalacticusFilter.clearCache()
        
        """
        self.cache = {}
        return
        
    def write(self,FILTER):
        """
        GalacticusFilter.write: Write a filter to an XML file.
        
        USAGE:  GalacticusFilter.write(FILTER)

              INPUT
                  FILTER  -- Instance of Filter class containing filter information.

        """
        FILTER.writeToFile()
        return
                    
    def load(self,filterName):
        """
        GalacticusFilter.load: Load a filter from an XML file in the Galacticus dataset repository.
                               
        USAGE: FILTER = GalacticusFilter.load(filterName)
        
           INPUT

                 filterName    -- Name of the filter. This will be used to identify the file. For
                                  example, 'SDSS_r' will instruct the class to search for a file
                                  'SDSS_r.xml' in the Galacticus datasets repository.

            OUTPUT
              
                 FILTER        -- Instance of the Filter class.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # First check whether we have this filter stored in memory
        if filterName in self.cache.keys():
            return self.cache[filterName]
        # Check if filter stored in datasets respository        
        if fnmatch.fnmatch(filterName.lower(),"*tophat*"):
            FILTER = self.TOPHATS.create(filterName,writeToFile=False)
        else:
            filterFile = self.DATA.search(filterName+".xml")
            FILTER = Filter()
            FILTER.loadFromFile(filterFile)
            FILTER.vegaOffset = self.VEGA.abVegaOffset(FILTER.transmission.wavelength,\
                                                           FILTER.transmission.transmission)
            self.cache[filterName] = FILTER
        return FILTER


