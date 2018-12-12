#! /usr/bin/env python

import os
import numpy as np
import copy
import unittest
from scipy.stats import norm
from scipy.integrate import romb
from scipy.interpolate import interp1d
from galacticus.fileFormats.xmlTree import xmlTree
from galacticus.data import GalacticusData
from galacticus.filters import computeEffectiveWavelength
from galacticus.filters import Filter,loadFilterFromFile
from galacticus.constants import speedOfLight,angstrom
from galacticus.constants import luminosityAB,luminositySolar

class TestFiltersInit(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.DATA = GalacticusData()
        return

    @classmethod
    def tearDownClass(self):
        tmpfile = self.DATA.dynamic+"filters/unitTestGaussianFilter.xml"
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        tmpfile = self.DATA.dynamic+"filters/GaussianFilter.xml"
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        return
        

    def createGaussianTransmission(self,wavelengths=np.linspace(4000,8000,200),
                                   loc=6000.0,scale=500.0):        
        transmission = np.zeros(len(wavelengths),dtype=[("wavelength",float),("transmission",float)])
        transmission["wavelength"] = wavelengths
        GAUSS = norm(loc=loc,scale=scale)        
        trans = GAUSS.pdf(wavelengths)
        trans /= trans.max()
        transmission["transmission"] = trans
        return transmission.view(np.recarray)

    def createGaussianFilter(self,wavelengths=np.linspace(4000,8000,200),loc=6000.0,scale=500.0):
        FILTER = Filter()
        FILTER.name = "GaussianFilter"
        FILTER.description = "A Gaussian transmission curve centered on "+\
            str(loc)+"A with width "+str(scale)+"A."
        FILTER.origin = "unitTestFilter function"
        FILTER.url = "None"
        trans = self.createGaussianTransmission(wavelengths=wavelengths,loc=loc,scale=scale)
        FILTER.transmission = trans
        FILTER.effectiveWavelength = computeEffectiveWavelength(trans["wavelength"],trans["transmission"])
        FILTER.vegaOffset = 0.1
        return FILTER
        
    def test_computeEffectiveWavelength(self):
        transmission = self.createGaussianTransmission()
        truth = np.sum(transmission["wavelength"]*transmission["transmission"])
        truth /= np.sum(transmission["transmission"])
        result = computeEffectiveWavelength(transmission["wavelength"],transmission["transmission"])
        self.assertEqual(result,truth)
        return

    def test_FilterReset(self):
        FILTER = self.createGaussianFilter()
        FILTER.reset()
        self.assertTrue(all([FILTER.__dict__[key] is None 
                             for key in FILTER.__dict__.keys()]))
        return
        
    def test_FilterSetEffectiveWavelength(self):
        FILTER = self.createGaussianFilter()
        eff = copy.copy(FILTER.effectiveWavelength)
        FILTER.setEffectiveWavelength()
        self.assertEqual(eff,FILTER.effectiveWavelength)
        return
    
    def test_FilterSetTransmission(self):
        FILTER = self.createGaussianFilter()
        transmission = self.createGaussianTransmission().view(np.recarray)
        transmission["transmission"] *= 10.0
        FILTER.setTransmission(transmission["wavelength"],transmission["transmission"])
        diff = np.fabs(transmission["wavelength"]-FILTER.transmission["wavelength"])
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        diff = np.fabs(transmission["transmission"]-FILTER.transmission["transmission"])
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        return
    
    def test_FilterInterpolate(self):
        wavelengths=np.linspace(4000,8000,200)
        loc=6000.0
        scale=500.0
        FILTER = self.createGaussianFilter(wavelengths=wavelengths,loc=loc,scale=scale)        
        GAUSS = norm(loc=loc,scale=scale)
        trans = GAUSS.pdf(wavelengths)
        MAX = trans.max()        
        # Test 1: wavelengths inside filter
        wave = np.array([4001.,5657.,6932.,7998.])
        data = FILTER.interpolate(wave)
        values = GAUSS.pdf(wave)/MAX
        diff = np.fabs(values-data)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        # Test 2: wavelengths outside filter
        wave = np.array([400.,3999.,8001.,99990.])
        data = FILTER.interpolate(wave)
        [self.assertEqual(d,0.0) for d in data]
        return

    def test_FilterIntegrate(self):
        wavelengths=np.linspace(4000,8000,200)
        loc=6000.0
        scale=500.0
        FILTER = self.createGaussianFilter(wavelengths=wavelengths,loc=loc,scale=scale)        
        kRomb = 10        
        # Manual calculation
        wavelengths = np.linspace(4000.0,8000.0,2**kRomb+1)
        transmission = self.createGaussianTransmission(wavelengths=wavelengths,loc=loc,scale=scale)
        deltaWavelength = wavelengths[1] - wavelengths[0]
        value = np.copy(transmission.transmission)/wavelengths**2
        value *= speedOfLight*luminosityAB/(angstrom*luminositySolar)
        result = romb(value,dx=deltaWavelength)
        # Filter function
        data = FILTER.integrate(kRomb=kRomb)
        # Compare and check differ by less than 0.1%
        diff = np.fabs(data-result)/result
        self.assertLessEqual(diff,0.001)
        return

    def test_FilterWriteToFile(self):
        FILTER0 = self.createGaussianFilter()        
        # Test 1: Provide filter file name
        tmpfile = self.DATA.dynamic+"filters/unitTestGaussianFilter.xml"
        if not os.path.exists(self.DATA.dynamic+"filters/"):
                os.makedirs(self.DATA.dynamic+"filters/")
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        FILTER0.writeToFile(tmpfile)
        self.assertTrue(os.path.exists(tmpfile))
        FILTER1 = Filter()
        FILTER1.loadFromFile(tmpfile)
        self.assertEqual(FILTER0.name,FILTER1.name)
        self.assertEqual(FILTER0.effectiveWavelength,FILTER1.effectiveWavelength)
        self.assertEqual(FILTER0.description,FILTER1.description)
        self.assertEqual(FILTER0.vegaOffset,FILTER1.vegaOffset)
        self.assertEqual(FILTER0.origin,FILTER1.origin)
        self.assertEqual(FILTER0.url,FILTER1.url)        
        diff = np.fabs(FILTER0.transmission.wavelength-FILTER1.transmission.wavelength)
        diff /= FILTER0.transmission.wavelength        
        [self.assertLessEqual(d,0.001) for d in diff]
        diff = np.fabs(FILTER0.transmission.transmission-FILTER1.transmission.transmission)
        diff /= FILTER0.transmission.transmission        
        [self.assertLessEqual(d,0.001) for d in diff]
        # Test 2: no name provided
        FILTER0 = self.createGaussianFilter()        
        ofile = self.DATA.dynamic+"filters/"+FILTER0.name+".xml"
        if os.path.exists(ofile):
            os.remove(ofile)
        FILTER0.writeToFile()
        self.assertTrue(os.path.exists(ofile))        
        return

    def test_FilterLoadFromFile(self):
        FILTER = Filter()
        filterFile = self.DATA.search("SDSS_r.xml")
        FILTER.loadFromFile(filterFile)
        self.assertIsNotNone(FILTER.name)
        self.assertEqual(FILTER.name,"SDSS r")
        self.assertIsNotNone(FILTER.description)
        self.assertIsNotNone(FILTER.origin)
        self.assertEqual(FILTER.origin,"Michael Blanton")
        self.assertIsNotNone(FILTER.effectiveWavelength)
        diff = np.fabs(FILTER.effectiveWavelength-6198.41999837059)
        self.assertLessEqual(diff,1.0e-6)
        self.assertIsNotNone(FILTER.vegaOffset)
        diff = np.fabs(FILTER.vegaOffset--0.139302055718797)
        self.assertLessEqual(diff,1.0e-6)
        # Manually load transmission
        TREE = xmlTree(file=filterFile)
        RESP = TREE.getElement("/filter/response")
        DATA = RESP.findall("datum")
        dtype = [("wavelength",float),("transmission",float)]
        transmission = np.zeros(len(DATA),dtype=dtype).view(np.recarray)
        for i,datum in enumerate(DATA):
            transmission["wavelength"][i] = float(datum.text.split()[0])
            transmission["transmission"][i] = float(datum.text.split()[1])
        diff = np.fabs(FILTER.transmission.wavelength-transmission.wavelength)
        diff /= transmission.wavelength
        [self.assertLessEqual(d,0.001) for d in diff]
        diff = np.copy(np.fabs(FILTER.transmission.transmission-transmission.transmission))        
        np.place(transmission.transmission,transmission.transmission==0.0,1.0)
        diff /= transmission.transmission
        [self.assertLessEqual(d,0.001) for d in diff]
        return


    def test_LoadFilterFromFile(self):
        filterFile = self.DATA.search("SDSS_r.xml")
        FILTER = loadFilterFromFile(filterFile)
        self.assertIsNotNone(FILTER.name)
        self.assertEqual(FILTER.name,"SDSS r")
        self.assertIsNotNone(FILTER.description)
        self.assertIsNotNone(FILTER.origin)
        self.assertEqual(FILTER.origin,"Michael Blanton")
        self.assertIsNotNone(FILTER.effectiveWavelength)
        diff = np.fabs(FILTER.effectiveWavelength-6198.41999837059)
        self.assertLessEqual(diff,1.0e-6)
        self.assertIsNotNone(FILTER.vegaOffset)
        diff = np.fabs(FILTER.vegaOffset--0.139302055718797)
        self.assertLessEqual(diff,1.0e-6)
        # Manually load transmission
        TREE = xmlTree(file=filterFile)
        RESP = TREE.getElement("/filter/response")
        DATA = RESP.findall("datum")
        dtype = [("wavelength",float),("transmission",float)]
        transmission = np.zeros(len(DATA),dtype=dtype).view(np.recarray)
        for i,datum in enumerate(DATA):
            transmission["wavelength"][i] = float(datum.text.split()[0])
            transmission["transmission"][i] = float(datum.text.split()[1])
        diff = np.fabs(FILTER.transmission.wavelength-transmission.wavelength)
        diff /= transmission.wavelength
        [self.assertLessEqual(d,0.001) for d in diff]
        diff = np.copy(np.fabs(FILTER.transmission.transmission-transmission.transmission))        
        np.place(transmission.transmission,transmission.transmission==0.0,1.0)
        diff /= transmission.transmission
        [self.assertLessEqual(d,0.001) for d in diff]
        return


        

if __name__ == "__main__":
    unittest.main()
        
