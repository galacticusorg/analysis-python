#! /usr/bin/env python

import sys
import numpy as np
from . import Filter
from .io import loadFilterFromFile,writeFilterToFile
from .vega import VegaOffset
from ..data import GalacticusData



class GalacticusFilter(Filter):
    
    def __init__(self,path=None,vbandFilterName="Buser_V",spectrumFile="A0V_Castelli.xml",\
                     verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(self.__class__.__name__,self).__init__(verbose=verbose)
        self._GalacticusData = GalacticusData(band=band,verbose=self.verbose)
        self._VegaOffset = VegaOffset(path=path,filterName=vbandFilterName,\
                                          spectrumFile=spectrumFile,verbose=self.verbose)
        self._filtersInMemory = {}
        return

    def clearMemory(self):
        self._filtersInMemory = {}
        return
        
    def create(self,name,response,description=None,effectiveWavelength=None,origin=None,\
                   url=None,vegaOffset=None,writeToFile=True,kRomberg=8,**kwargs):
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
        return FILTER

    def write(self,FILTER):
        writeFilterToFile(FILTER,self._GalacticusData.dynamic)
        return
                    
    def load(self,filterName,keepInMemory=False,kRomberg=8,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # First check whether we have this filter stored in memory
        if filterName in self._filtersInMemory.keys():
            return self._filtersInMemory[filterName]
        # Check if filter stored in datasets respository        
        if filterName not in self.filters.keys():
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
