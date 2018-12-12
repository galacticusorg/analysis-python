#! /usr/bin/env python

import os
import numpy as np
import unittest
from unittest.mock import patch
from galacticus.data import GalacticusData
from galacticus.filters.topHats import getTransmissionCurve,TopHat
from galacticus.errors import ParseError

class TestTopHat(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.DATA = GalacticusData()
        return

    def test_TopHatGetFilterSizeAdaptiveResolution(self):
        name = "adaptiveResolutionTopHat_10020.0_500.0"
        center,width = TopHat.getFilterSizeAdaptiveResolution(name)
        self.assertEqual(center,10020.0)
        self.assertEqual(width,500.0)
        name = "adaptiveResolutionTopHat_10020_500"
        center,width = TopHat.getFilterSizeAdaptiveResolution(name)
        self.assertEqual(center,10020.0)
        self.assertEqual(width,500.0)
        with self.assertRaises(ParseError):
            name = "adaptiveResolutionTopHat_10020_-500"
            center,width = TopHat.getFilterSizeAdaptiveResolution(name)
            name = "adaptiveResolutionTopHat_-10020_500"
            center,width = TopHat.getFilterSizeAdaptiveResolution(name)
            name = "fixedResolutionTopHat_10020_500"
            center,width = TopHat.getFilterSizeAdaptiveResolution(name)
        return
        
    def test_TopHatGetFilterSizeFixedResolution(self):
        with self.assertRaises(ParseError):            
            name = "adaptiveResolutionTopHat_10020.0_500.0"
            center,width = TopHat.getFilterSizeFixedResolution(name)
            name = "fixedResolutionTopHat_-10020.0_500.0"
            center,width = TopHat.getFilterSizeFixedResolution(name)
            name = "fixedResolutionTopHat_10020.0_-500.0"
            center,width = TopHat.getFilterSizeFixedResolution(name)            
        name = "fixedResolutionTopHat_10020.0_50.0"
        center,width = TopHat.getFilterSizeFixedResolution(name)        
        self.assertEqual(center,10020.0)
        # Manual calculation for width
        wavelengthCentral = 10020.0
        resolution = 50.0
        wavelengthRatio = (np.sqrt(4.0*resolution**2+1.0)+1.0)/(np.sqrt(4.0*resolution**2+1.0)-1.0)
        wavelengthMinimum = wavelengthCentral*(np.sqrt(4.0*resolution**2+1.0)-1.0)/2.0/resolution
        wavelengthMinimum /= wavelengthRatio
        wavelengthMaximum = wavelengthCentral*(np.sqrt(4.0*resolution**2+1.0)+1.0)/2.0/resolution
        wavelengthMaximum /= wavelengthRatio
        wavelengthWidth = wavelengthMaximum - wavelengthMinimum
        self.assertEqual(width,wavelengthWidth)
        return

    def test_TopHatGetFilterSize(self):
        with self.assertRaises(ParseError):            
            names = ["adaptiveResolutionTopHat_10020_-500",
                     "adaptiveResolutionTopHat_-10020_500",
                     "fixedResolutionTopHat_-10020.0_500.0",
                     "fixedResolutionTopHat_10020.0_-500.0"]  
            for name in names:
                center,width = TopHat.getFilterSize(name)
        # Test adaptive width filter
        name = "adaptiveResolutionTopHat_10020_500"
        center,width = TopHat.getFilterSize(name)
        self.assertEqual(center,10020.0)
        self.assertEqual(width,500.0)
        # Test fixed width filter
        name = "fixedResolutionTopHat_10020.0_50.0"
        center,width = TopHat.getFilterSize(name)        
        self.assertEqual(center,10020.0)
        # Manual calculation for width
        wavelengthCentral = 10020.0
        resolution = 50.0
        wavelengthRatio = (np.sqrt(4.0*resolution**2+1.0)+1.0)/(np.sqrt(4.0*resolution**2+1.0)-1.0)
        wavelengthMinimum = wavelengthCentral*(np.sqrt(4.0*resolution**2+1.0)-1.0)/2.0/resolution
        wavelengthMinimum /= wavelengthRatio
        wavelengthMaximum = wavelengthCentral*(np.sqrt(4.0*resolution**2+1.0)+1.0)/2.0/resolution
        wavelengthMaximum /= wavelengthRatio
        wavelengthWidth = wavelengthMaximum - wavelengthMinimum
        self.assertEqual(width,wavelengthWidth)                
        return

    def test_TopHatCreate(self):
        path = "galacticus.filters.vega.Vega.abVegaOffset"
        with patch(path) as mocked_offset:
            mocked_offset.return_value = 0.1
            filterName = "adaptiveResolutionTopHat_10020_500"
            size = 1000
            bufferFrac = 0.1
            TOP = TopHat()
            center,width = TOP.getFilterSize(filterName)                       
            FILTER = TOP.create(filterName,writeToFile=False,transmissionSize=size,
                                   edgesBufferFraction=bufferFrac)
            self.assertEqual(FILTER.name,filterName)
            self.assertEqual(FILTER.origin,"Galacticus source code")
            self.assertEqual(FILTER.url,"None")
            self.assertEqual(FILTER.vegaOffset,0.1)
            # Compute points of interest
            fraction = 0.5 + bufferFrac
            lowerLimit = center - width*fraction
            upperLimit = center + width*fraction
            lowerEdge = center - width/2.0
            upperEdge = center + width/2.0
            # Check filter matches these specifications
            self.assertEqual(FILTER.transmission.wavelength.min(),lowerLimit)
            self.assertEqual(FILTER.transmission.wavelength.max(),upperLimit)
            mask = np.logical_and(FILTER.transmission.wavelength>=lowerEdge,FILTER.transmission.wavelength<=upperEdge)
            self.assertTrue(all(FILTER.transmission.transmission[mask]==1.0))
            mask = FILTER.transmission.wavelength < lowerEdge
            self.assertTrue(all(FILTER.transmission.transmission[mask]==0.0))
            mask = FILTER.transmission.wavelength > upperEdge
            self.assertTrue(all(FILTER.transmission.transmission[mask]==0.0))
            # Check effective wavelength
            truth = np.sum(FILTER.transmission["wavelength"]*FILTER.transmission["transmission"])
            truth /= np.sum(FILTER.transmission["transmission"])
            self.assertEqual(FILTER.effectiveWavelength,truth)
            # Check writing to file
            ofile = self.DATA.dynamic+"/filters/"+filterName+".xml"
            if not os.path.exists(self.DATA.dynamic+"filters/"):
                os.makedirs(self.DATA.dynamic+"filters/")
            if os.path.exists(ofile):
                os.remove(ofile)
            FILTER = TOP.create(filterName,writeToFile=True,transmissionSize=size,
                                edgesBufferFraction=bufferFrac)
            self.assertTrue(os.path.exists(ofile))
            if os.path.exists(ofile):
                os.remove(ofile)            
        return


class TestGetTransmissionCurve(unittest.TestCase):
    
    def test_getTransmissionCurve(self):
        center = 1000.0
        width = 500.0
        size = 101
        bufferFrac = 0.1
        transmission = getTransmissionCurve(center,width,transmissionSize=size,
                                            edgesBufferFraction=bufferFrac)
        # Compute points of interest
        fraction = 0.5 + bufferFrac
        lowerLimit = center - width*fraction
        upperLimit = center + width*fraction
        lowerEdge = center - width/2.0
        upperEdge = center + width/2.0
        # Check filter matches these specifications
        self.assertEqual(transmission.wavelength.min(),lowerLimit)
        self.assertEqual(transmission.wavelength.max(),upperLimit)
        mask = np.logical_and(transmission.wavelength>=lowerEdge,transmission.wavelength<=upperEdge)
        self.assertTrue(all(transmission.transmission[mask]==1.0))
        mask = transmission.wavelength < lowerEdge
        self.assertTrue(all(transmission.transmission[mask]==0.0))
        mask = transmission.wavelength > upperEdge
        self.assertTrue(all(transmission.transmission[mask]==0.0))
        return

if __name__ == "__main__":
    unittest.main()


        
