#! /usr/bin/env python

import __future__
import sys
import numpy as np
import xml.etree.ElementTree as ET
from ..fileFormats.xmlTree import xmlTree
from ..data import GalacticusData

def computeEffectiveWavelength(wavelength,transmission):
    """
    computeEffectiveWavelength: Compute the effective wavelength for a filter transmission.

    USAGE: effectiveWavelength = computeEffectiveWavelength(wavelength,transmission)

       INPUT 
            wavelength   -- Numpy array of wavelengths.
            transmission -- Numpy array of filter transmssion curve.

       OUTPUT
            effectiveWavelength -- Effactive wavelength for transmission curve.
    """
    return np.sum(wavelength*transmission)/np.sum(transmission)


class Filter(object):

    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.file = None
        self.transmission = None
        self.description = None
        self.effectiveWavelength = None
        self.vegaOffset = None
        self.name = None
        self.origin = None
        self.url = None
        return

    def reset(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.file = None
        self.transmission = None
        self.description = None
        self.effectiveWavelength = None
        self.vegaOffset = None
        self.name = None
        self.origin = None
        self.url = None
        return

    def setEffectiveWavelength(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.transmission is None:
            raise ValueError(funcname+"(): no filter transmission has been set.")
        self.effectiveWavelength = computeEffectiveWavelength(self.transmission.wavelength,\
                                                                  self.transmission.transmission)
        return

    def setTransmission(self,wavelength,response):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store transmission
        if len(wavelength) != len(response):
            raise ValueError(funcname+"(): wavelength and response arrays are different length.")
        dtype = [("wavelength",float),("transmission",float)]
        self.transmission = np.zeros(len(wavelength),dtype=dtype).view(np.recarray)
        self.transmission.wavelength = wavelength
        self.transmission.transmission = response
        self.setEffectiveWavelength()
        return

    def loadFromFile(self,filterFile):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.reset()
        self.file = filterFile
        # Load file into XML tree
        TREE = xmlTree(file=filterFile)
        # Load filter attributes
        self.name = TREE.getElement("/filter/name").text
        if TREE.elementExists("/filter/description"): 
            self.description = TREE.getElement("/filter/description").text
        if TREE.elementExists("/filter/origin"): 
            self.origin = TREE.getElement("/filter/origin").text
        if TREE.elementExists("/filter/url"): 
            self.url = TREE.getElement("/filter/url").text
        # Load filter transmission
        RESP = TREE.getElement("/filter/response")
        DATA = RESP.findall("datum")
        dtype = [("wavelength",float),("transmission",float)]
        self.transmission = np.zeros(len(DATA),dtype=dtype).view(np.recarray)
        [("wavelength",float),("transmission",float)]        
        for i,datum in enumerate(DATA):
            self.transmission["wavelength"][i] = float(datum.text.split()[0])
            self.transmission["transmission"][i] = float(datum.text.split()[1])        
        # Load effective wavelength
        if TREE.elementExists("/filter/effectiveWavelength"): 
            self.effectiveWavelength = TREE.getElement("/filter/effectiveWavelength").text
        else:
            self.setEffectiveWavelength()
        # Load AB/Vega offset (if present)
        if TREE.elementExists("/filter/vegaOffset"):
            self.vegaOffset = TREE.getElement("/filter/vegaOffset").text
        del TREE
        return 

    def writeToFile(self,outFile=None):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if outFile is None:
            DATA = GalacticusData(verbose=False)
            os.makedirs(DATA.dynamic+"/filters/")
            outFile = DATA.dyamic+"/filters/"+self.name+".xml"
        TREE = xmlTree(root="filter")
        TREE.createElement("/filter","description",text=str(self.description))
        TREE.createElement("/filter","name",text=self.name)
        TREE.createElement("/filter","origin",text=str(self.origin))        
        TREE.createElement("/filter","url",text=str(self.url))
        TREE.createElement("/filter","response")
        if self.transmission is None:
            raise ValueError(funcname+"(): filter has no transmission curve!")
        for i in range(len(self.transmission.wavelength)):
            datum = "{0:7.3f}      {1:9.7e}".format(self.transmission.wavelength[i],\
                                                   self.transmission.transmission[i])
            TREE.createElement("/filter/response","datum",text=datum)
        TREE.createElement("/filter","effectiveWavelength",text=str(self.effectiveWavelength))
        TREE.createElement("/filter","vegaOffset",text=str(self.vegaOffset))
        TREE.writeToFile(outFile)
        return

                
def loadFilterFromFile(filterFile):
    FILTER = Filter()
    FILTER.loadFromFile(filterFile)
    return FILTER

