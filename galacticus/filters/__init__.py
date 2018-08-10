#! /usr/bin/env python

import __future__
import sys,os
import numpy as np
import unittest
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


class UnitTest(unittest.TestCase):

    def testLoadFilter(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Filter: "+funcname)
        print("Creating a Filter() class instance")
        FILTER = Filter()
        print("Testing loading a filter from a file")
        DATA = GalacticusData(verbose=True)
        filterFile = DATA.search("SDSS_r.xml")
        FILTER.loadFromFile(filterFile)
        self.assertIsNotNone(FILTER.name)
        self.assertIsNotNone(FILTER.description)
        self.assertIsNotNone(FILTER.origin)
        self.assertIsNotNone(FILTER.effectiveWavelength)
        self.assertIsNotNone(FILTER.vegaOffset)
        self.assertIsNotNone(FILTER.transmission)
        print("TEST COMPLETE")
        print("\n")
    
    def testCreateFilter(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Filter: "+funcname)
        print("Creating a Filter() class instance")
        FILTER = Filter()
        print(" Create an example Gaussian filter")
        from scipy.stats import norm
        loc = 6000.0
        scale = 500.0
        print(" mean = "+str(loc)+" stdev = "+str(scale))
        GAUSS = norm(loc=loc,scale=scale)
        wave = np.linspace(4000,8000,200)
        trans = GAUSS.pdf(wave)
        trans /= trans.max()
        print("Storing filter attributes")
        FILTER = Filter()
        FILTER.name = "GaussianFilter"
        FILTER.description = "A Gaussian transmission curve centered on "+str(loc)+"A with width "+str(scale)+"A."
        FILTER.origin = "unitTestFilter function"
        FILTER.url = "None"
        response = np.zeros(len(wave),dtype=[("wavelength",float),("transmission",float)]).view(np.recarray)
        response["wavelength"] = wave
        response["transmission"] = trans
        FILTER.transmission = response
        FILTER.setEffectiveWavelength()
        self.assertIsNotNone(FILTER.description)
        self.assertIsNotNone(FILTER.origin)
        self.assertIsNotNone(FILTER.url)
        self.assertIsNotNone(FILTER.effectiveWavelength)
        self.assertIsNotNone(FILTER.transmission)
        print("TEST COMPLETE")
        print("\n")

    def testWriteFilter(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Filter: "+funcname)
        print("Creating a Filter() class instance")
        FILTER = Filter()
        print(" Create an example Gaussian filter")
        from scipy.stats import norm
        loc = 6000.0
        scale = 500.0
        print(" mean = "+str(loc)+" stdev = "+str(scale))
        GAUSS = norm(loc=loc,scale=scale)
        wave = np.linspace(4000,8000,200)
        trans = GAUSS.pdf(wave)
        trans /= trans.max()
        print("Storing filter attributes")
        FILTER = Filter()
        FILTER.name = "GaussianFilter"
        FILTER.description = "A Gaussian transmission curve centered on "+str(loc)+"A with width "+str(scale)+"A."
        FILTER.origin = "unitTestFilter function"
        FILTER.url = "None"
        response = np.zeros(len(wave),dtype=[("wavelength",float),("transmission",float)]).view(np.recarray)
        response["wavelength"] = wave
        response["transmission"] = trans
        FILTER.transmission = response
        FILTER.setEffectiveWavelength()
        print("Writing Gaussian filter to file")        
        tmpfile = "gaussianFilter.xml"
        FILTER.writeToFile(tmpfile)
        print("Re-reading Gaussian filter from file")
        FILTER2 = Filter()
        FILTER2.loadFromFile(tmpfile)
        print("Checking attributes read from file are consistent")
        for key in ["name","origin","url","description"]:
            self.assertEqual(FILTER.__dict__[key],FILTER2.__dict__[key])        
        self.assertTrue(np.fabs(FILTER.effectiveWavelength-FILTER2.effectiveWavelength)<1.0e-4)
        for i in range(len(FILTER.transmission.wavelength)):
            subPercent = FILTER.transmission.wavelength[i]*0.001
            diff = np.fabs(FILTER.transmission.wavelength[i]-FILTER2.transmission.wavelength[i])
            self.assertTrue(diff<subPercent)
            subPercent = FILTER.transmission.transmission[i]*0.001
            diff = np.fabs(FILTER.transmission.transmission[i]-FILTER2.transmission.transmission[i])
            self.assertTrue(diff<subPercent)
        os.remove(tmpfile)
        print("TEST COMPLETE")
        print("\n")
        return
        


