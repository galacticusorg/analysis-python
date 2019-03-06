#! /usr/bin/env python

import __future__
import sys,os
import numpy as np
import unittest
import xml.etree.ElementTree as ET
from scipy.stats import norm
from scipy.integrate import romb
from scipy.interpolate import interp1d
from ..fileFormats.xmlTree import xmlTree
from ..data import GalacticusData
from ..constants import speedOfLight,angstrom
from ..constants import luminosityAB,luminositySolar

def computeEffectiveWavelength(wavelength,transmission):
    """
    computeEffectiveWavelength: Compute the effective wavelength for a filter transmission.

    USAGE: effectiveWavelength = computeEffectiveWavelength(wavelength,transmission)

       INPUT 
            wavelength   -- Numpy array of wavelengths.
            transmission -- Numpy array of filter transmssion curve.

       OUTPUT
            effectiveWavelength -- Effective wavelength for transmission curve.
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
        self.effectiveWavelength = computeEffectiveWavelength(self.transmission.wavelength,
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
        for i,datum in enumerate(DATA):
            self.transmission["wavelength"][i] = float(datum.text.split()[0])
            self.transmission["transmission"][i] = float(datum.text.split()[1])        
        # Load effective wavelength
        if TREE.elementExists("/filter/effectiveWavelength"): 
            self.effectiveWavelength = float(TREE.getElement("/filter/effectiveWavelength").text)
        else:
            self.setEffectiveWavelength()
        # Load AB/Vega offset (if present)
        if TREE.elementExists("/filter/vegaOffset"):
            self.vegaOffset = TREE.getElement("/filter/vegaOffset").text
            try:
                self.vegaOffset = float(self.vegaOffset)
            except ValueError:
                pass
                
        del TREE
        return 

    def writeToFile(self,outFile=None):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if outFile is None:
            DATA = GalacticusData(verbose=False)
            if not os.path.exists(DATA.dynamic+"/filters/"):
                os.makedirs(DATA.dynamic+"/filters/")
            outFile = DATA.dynamic+"/filters/"+self.name+".xml"
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

    def interpolate(self,wavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        TRANSMISSION = interp1d(self.transmission.wavelength,\
                                    self.transmission.transmission,\
                                    kind='cubic',\
                                    fill_value=0.0,bounds_error=False)
        return TRANSMISSION(wavelength)

    def integrate(self,kRomb=10):
        # Integrate a zero-magnitude AB source under the filter
        wavelengths = np.linspace(self.transmission.wavelength.min(),\
                                  self.transmission.wavelength.max(),\
                                      2**kRomb+1)
        deltaWavelength = wavelengths[1] - wavelengths[0]
        transmission = self.interpolate(wavelengths)
        transmission /= wavelengths**2
        transmission *= speedOfLight*luminosityAB/(angstrom*luminositySolar)
        return romb(transmission,dx=deltaWavelength)                
    

def loadFilterFromFile(filterFile):
    FILTER = Filter()
    FILTER.loadFromFile(filterFile)
    return FILTER  


