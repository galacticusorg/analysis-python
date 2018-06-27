#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from . import Filter
from .io import loadFilterFromFile,writeFilterToFile
from .vega import VegaOffset
from ..data import GalacticusData



class GalacticusFilter(Filter):    
    """
    GalacticusFilter: Class to manage reading and writing of filter transmissions used 
                      with Galacticus.

    USAGE: GalFilter = GalacticusFilter([path=None],[vbandFilterName=Buser_V],\
                                        [spectrumFile=A0V_Castelli.xml],[verbose=False])

        INTPUT 
                    path     -- Path to datasets repository. If None, will search for path in 
                                environment variables (stored as 'GALACTICUS_DATASETS'). 
                                [Default=None] 
             vbandFilterName -- Name of file containing V-band transmission curve. It is 
                                assumed that this file is either in the dynamic or static
                                subdirectories of the datasets repository. If not file name
                                is specified the class will assume that the transmission
                                is the V-band filter 'Buser_V'. [Default=Buser_V]
                spectrumFile -- Name of file containing Vega spectral information. It is 
                                assumed that this file is either in the dynamic or static 
                                subdirectories of the datasets repository. If not file name is
                                specified the class will assume that the spectrum is stored
                                in a file named 'A0V_Castelli.xml'. [Default=A0V_Castelli.xml]
                verbose      -- Increase verbosity. [Default=False]


          OUTPUT
                  GalFilter  -- Class object instance.


    Note: Upon initialization the class will create an instance of the VegaSpectrum and 
    VegaOffset classes. These are necessary for computing AB/Vega filter offsets.

    Functions:

             clearMemory:  Clear all isntances of filters stored in memory.
             create: Create an instance of the Filter class with specified information.
             write: Write Filter class instance to an XML file.
             load: Load a filter from an XML file.

    """    
    def __init__(self,path=None,vbandFilterName="Buser_V",spectrumFile="A0V_Castelli.xml",\
                     verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(self.__class__,self).__init__()
        self.verbose = verbose
        self._GalacticusData = GalacticusData(path=path,verbose=self.verbose)
        self._VegaOffset = VegaOffset(path=path,filterName=vbandFilterName,\
                                          spectrumFile=spectrumFile,verbose=self.verbose)
        self._filtersInMemory = {}
        return

    def clearMemory(self):
        """
        GalacticusFilter.clearMemory: Clear filter objects stored in memory.
          
        USAGE   GalacticusFilter.clearMemory()
        
        """
        self._filtersInMemory = {}
        return
        
    def create(self,name,response,description=None,effectiveWavelength=None,origin=None,\
                   url=None,vegaOffset=None,writeToFile=True,keepInMemory=False,kRomberg=8,\
                   **kwargs):
        """
        GalacticusFilter.create: Create a Filter class instance and, optionally, write 
                                 the filter information to a file.


        USAGE: FILTER = GalacticusFilter.create(name,response,[description=None],\
                                                [effectiveWavelength=None],[origin=None],\
                                                [url=None],[vegaOffset=None],[writeToFile=True],\
                                                [keepInMemory=False],[kRomberg=8],[**kwargs])
                                       
             INPUT
             
                 name                 -- Name to use for the filter. Note that if the filter is
                                         written to disk, this name will be used as the output
                                         file name.                                         
                 response             -- Numpy record array storing filter transmission curve. Fields
                                         in the record array should be 'wavelength' and 'response'.
                 description          -- Optional additional information for the filter. [Default=None]
                 effectiveWavelength  -- Effective wavelength for the filter. If not specified, will
                                         be computed by the class. [Default=None]
                 origin               -- Optional information regarding origin of the transmission curve.
                                         [Default=None]
                 url                  -- Optional information regarding URL of the transmission curve.
                                         [Default=None]
                 vegaOffset           -- Value for AB/Vega offset for this filter. If not specified, will
                                         be computed by the class. [Default=None] 
                 writeToFile          -- Flag indicating filter is to be written to an XML file. All 
                                         user-created filters are by default written to the dynamic/filters 
                                         subdirectory of the Galacticus datasets repository.                
                 keepInMemory         -- Store filter in memory for future use. [Default=False]
                 kRomberg             -- Number of k-nodes for Romberg integration. [Default=8]
                 **kwrgs              -- Keywords arguments to pass to scipy.interpolate.interp1d.

              OUTPUT
              
                 FILTER                -- Instance of the Filter class.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        FILTER = Filter()
        FILTER.name = name
        FILTER.setTransmission(response["wavelength"],response["response"])
        if effectiveWavelength is not None:
            FILTER.effectiveWavelength = effectiveWavelength
        else:
            FILTER.setEffectiveWavelength()
        if vegaOffset is not None:
            FILTER.vegaOffset = vegaOffset
        else:
            FILTER.vegaOffset = self._VegaOffset.computeOffset(response["wavelength"],\
                                                              response["response"],\
                                                              kRomberg=8,**kwargs)
        FILTER.origin = origin
        FILTER.description = description
        FILTER.url = url
        if writeToFile:
            self.write(FILTER)
        if keepInMemory:
             self._filtersInMemory[filterName] = FILTER
        return FILTER

    def write(self,FILTER):
        """
        GalacticusFilter.write: Write a filter to an XML file.
        
        USAGE:  GalacticusFilter.write(FILTER)

              INPUT
                  FILTER  -- Instance of Filter class containing filter information.

        """
        writeFilterToFile(FILTER,self._GalacticusData.dynamic+"filters/",verbose=self.verbose)
        return
                    
    def load(self,filterName,keepInMemory=False,kRomberg=8,**kwargs):        
        """
        GalacticusFilter.load: Load a filter from an XML file in the Galacticus dataset repository.
                               
        USAGE: FILTER = GalacticusFilter.load(filterName,[keepInMemory=False],[kRomberg=8],[**kwargs])
        
           INPUT

                 filterName    -- Name of the filter. This will be used to identify the file. For
                                  example, 'SDSS_r' will instruct the class to search for a file
                                  'SDSS_r.xml' in the Galacticus datasets repository.
                 keepInMemory  -- Store filter in memory for future use. [Default=False]
                 kRomberg      -- Number of k-nodes for Romberg integration. [Default=8]
                 **kwrgs       -- Keywords arguments to pass to scipy.interpolate.interp1d.

            OUTPUT
              
                 FILTER        -- Instance of the Filter class.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # First check whether we have this filter stored in memory
        if filterName in self._filtersInMemory.keys():
            return self._filtersInMemory[filterName]
        # Check if filter stored in datasets respository        
        if fnmatch.fnmatch(filterName.lower(),"*tophat*") or fnmatch.fnmatch(filterName,"*emissionLine*"):
            FILTER = None
            #TOPHAT = TopHat()
            #FILTER = TOPHAT(filterName,**kwargs)
        else:
            filterFile = self._GalacticusData.search(filterName+".xml")
            FILTER = loadFilterFromFile(filterFile)
            FILTER.vegaOffset = self._VegaOffset.computeOffset(FILTER.transmission.wavelength,\
                                                                   FILTER.transmission.transmission,\
                                                                   kRomberg=8,**kwargs)
        if keepInMemory:
             self._filtersInMemory[filterName] = FILTER
        return FILTER
