#! /usr/bin/env python

import sys,os
import numpy as np
import copy
import unittest
from scipy.interpolate import RegularGridInterpolator
from galacticus.dust.CompendiumTable import CompendiumTable

class TestCompendiumTable(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Intitialize class
        self.COMP = CompendiumTable()
        self.COMP.load()
        return

    def setUp(self):
        # Generate some random data
        N = 100
        minInc = self.COMP.inclinationTable.min()
        maxInc = self.COMP.inclinationTable.max()
        self.inclination = np.random.rand(N)*(maxInc-minInc) + minInc
        maxOD = self.COMP.opticalDepthTable.max()
        self.opticalDepth = np.random.rand(N)*maxOD
        minWave = self.COMP.wavelengthTable.min()
        maxWave = self.COMP.wavelengthTable.max()
        self.wavelength = np.random.rand(N)*(maxWave-minWave) + minWave
        minRad = self.COMP.spheroidScaleRadialTable.min()
        maxRad = self.COMP.spheroidScaleRadialTable.max()
        self.spheroidRadius = np.random.rand(N)*(maxRad-minRad) + minRad
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        del self.COMP
        return

    def test_CompendiumTableLoad(self):
        # Test of CompendiumTable.load()
        self.COMP.wavelengthTable = None
        self.COMP.inclinationTable = None
        self.COMP.opticalDepthTable = None
        self.COMP.spheroidScaleRadialTable = None
        self.COMP.attenuationDiskTable = None
        self.COMP.attenuationSpheroidTable = None
        self.COMP.extrapolationDiskTable = None
        self.COMP.extrapolationSpheroidTable = None
        self.COMP.opacity = None
        self.COMP.load()
        for attr in [self.COMP.wavelengthTable,
                     self.COMP.inclinationTable,
                     self.COMP.opticalDepthTable,
                     self.COMP.spheroidScaleRadialTable,
                     self.COMP.attenuationDiskTable,
                     self.COMP.attenuationSpheroidTable,
                     self.COMP.extrapolationDiskTable,
                     self.COMP.extrapolationSpheroidTable]:
            self.assertEqual(type(attr),np.ndarray)
        shape = (len(self.COMP.wavelengthTable),\
                     len(self.COMP.inclinationTable),\
                     len(self.COMP.opticalDepthTable))
        self.assertEqual(self.COMP.attenuationDiskTable.shape,shape)
        shape = (len(self.COMP.wavelengthTable),\
                     len(self.COMP.inclinationTable),\
                     len(self.COMP.opticalDepthTable),\
                     len(self.COMP.spheroidScaleRadialTable))
        self.assertEqual(self.COMP.attenuationSpheroidTable.shape,shape)
        shape = (2,len(self.COMP.wavelengthTable),\
                     len(self.COMP.inclinationTable))
        self.assertEqual(self.COMP.extrapolationDiskTable.shape,shape)
        shape = (2,len(self.COMP.wavelengthTable),\
                     len(self.COMP.inclinationTable),\
                     len(self.COMP.spheroidScaleRadialTable))
        self.assertEqual(self.COMP.extrapolationSpheroidTable.shape,shape)
        self.assertEqual(type(self.COMP.opacity),np.float64)
        return

    def test_CompendiumTableTablesLoaded(self):
        # Test of CompendiumTable.tablesLoaded()
        self.COMP.wavelengthTable = None
        self.assertFalse(self.COMP.tablesLoaded())
        self.COMP.load()
        self.assertTrue(self.COMP.tablesLoaded())
        return

    def test_CompendiumTableLoadOpacity(self):
        # Test of CompendiumTable.loadOpacity()
        self.COMP.opacity = None
        self.COMP.loadOpacity()
        self.assertEqual(type(self.COMP.opacity),np.float64)
        return

    def test_CompendiumTableResetInterpolators(self):
        # Test CompendiumTable.resetInterpolators()
        self.COMP.resetInterpolators()
        self.assertIsNone(self.COMP.interpolator)
        self.assertIsNone(self.COMP.extrapolator0)
        self.assertIsNone(self.COMP.extrapolator1)
        return

    def test_CompendiumTableAssertSpheroidRadialScaleInRange(self):        
        # Test of CompendiumTable.assertSpheroidRadialScaleInRange
        self.assertTrue(self.COMP.assertSpheroidRadialScaleInRange(self.spheroidRadius))
        with self.assertRaises(RuntimeError):            
            self.spheroidRadius[0] = self.COMP.spheroidScaleRadialTable[0] - 9999.9
            self.COMP.assertSpheroidRadialScaleInRange(self.spheroidRadius)
            self.spheroidRadius[0] = self.COMP.spheroidScaleRadialTable[1]
            self.spheroidRadius[-1] = self.COMP.spheroidScaleRadialTable[-1] + 999.9
            self.COMP.assertSpheroidRadialScaleInRange(self.spheroidRadius)
        return

    def test_CompendiumTableAssertInclinationInRange(self):
        # Test of CompendiumTable.assertInclinationInRange
        self.assertTrue(self.COMP.assertInclinationInRange(self.inclination))
        with self.assertRaises(RuntimeError):
             self.COMP.assertInclinationInRange(np.array([-1.0]))
             self.COMP.assertInclinationInRange(np.array([91.0]))
             self.COMP.assertInclinationInRange(np.array([self.COMP.inclinationTable[0]*0.99]))
             self.COMP.assertInclinationInRange(np.array([self.COMP.inclinationTable[-1]*1.1]))
        return

    def test_CompendiumTableAssertWavelengthInRange(self):
        # Test of CompendiumTable.assertWavelengthInRange
        self.assertTrue(self.COMP.assertWavelengthInRange(self.wavelength))
        with self.assertRaises(RuntimeError):
             self.COMP.assertWavelengthInRange(np.array([self.COMP.wavelengthTable[0]*0.99]))
             self.COMP.assertWavelengthInRange(np.array([self.COMP.wavelengthTable[-1]*1.1]))             
        return

    def test_CompendiumTableAssertOpticalDepthInRange(self):
        # Test of CompendiumTable.assertOpticalDepthInRange
        self.assertTrue(self.COMP.assertOpticalDepthInRange(self.opticalDepth))
        self.COMP.extrapolateOpticalDepth = True
        self.opticalDepth[-1] = self.COMP.opticalDepthTable[-1]*1.1
        self.assertTrue(self.COMP.assertOpticalDepthInRange(self.opticalDepth))
        self.COMP.extrapolateOpticalDepth = False
        with self.assertRaises(RuntimeError):
             self.COMP.assertOpticalDepthInRange(np.array([self.COMP.opticalDepthTable[0]*0.99]))
             self.COMP.assertOpticalDepthInRange(np.array([self.COMP.opticalDepthTable[-1]*1.1]))                          
        return

    def test_CompendiumTableGetInterpolationMask(self):
        # Testing of CompendiumTable.getInterpolationMask
        self.COMP.extrapolateOpticalDepth = False
        self.assertTrue(all(self.COMP.getInterpolationMask(self.opticalDepth)))
        self.COMP.extrapolateOpticalDepth = True
        self.assertTrue(all(self.COMP.getInterpolationMask(self.opticalDepth)))
        mask = np.random.rand(len(self.opticalDepth))<0.1
        self.opticalDepth[mask] = self.COMP.opticalDepthTable[-1]*100.0
        MASK = self.COMP.getInterpolationMask(self.opticalDepth)
        [self.assertEqual(m,M) for m,M in zip(np.invert(mask),MASK)]
        return

    def test_CompendiumTableGetExtrapolationMask(self):
        # Testing of CompendiumTable.getInterpolationMask
        self.COMP.extrapolateOpticalDepth = False
        self.assertTrue(all(np.invert(self.COMP.getExtrapolationMask(self.opticalDepth))))
        self.COMP.extrapolateOpticalDepth = True
        self.assertTrue(all(np.invert(self.COMP.getExtrapolationMask(self.opticalDepth))))
        mask = np.random.rand(len(self.opticalDepth))<0.1
        self.opticalDepth[mask] = self.COMP.opticalDepthTable[-1]*100.0
        MASK = self.COMP.getExtrapolationMask(self.opticalDepth)
        [self.assertEqual(m,M) for m,M in zip(mask,MASK)]
        return

    def test_CompendiumTableBuildDiskInterpolators(self):
        # Testing of CompendiumTable.buildDiskInterpolators
        self.COMP.resetInterpolators()
        self.COMP.buildDiskInterpolators()
        self.assertIsInstance(self.COMP.interpolator,RegularGridInterpolator)
        self.assertIsInstance(self.COMP.extrapolator0,RegularGridInterpolator)
        self.assertIsInstance(self.COMP.extrapolator1,RegularGridInterpolator)
        self.assertEqual(len(self.COMP.interpolator.grid),3)
        self.assertEqual(len(self.COMP.extrapolator0.grid),2)
        self.assertEqual(len(self.COMP.extrapolator1.grid),2)
        return

    def test_CompendiumTableBuildSpheroidInterpolators(self):
        # Testing of CompendiumTable.buildSpheroidInterpolators
        self.COMP.resetInterpolators()
        self.COMP.buildSpheroidInterpolators()
        self.assertIsInstance(self.COMP.interpolator,RegularGridInterpolator)
        self.assertIsInstance(self.COMP.extrapolator0,RegularGridInterpolator)
        self.assertIsInstance(self.COMP.extrapolator1,RegularGridInterpolator)
        self.assertEqual(len(self.COMP.interpolator.grid),4)
        self.assertEqual(len(self.COMP.extrapolator0.grid),3)
        self.assertEqual(len(self.COMP.extrapolator1.grid),3)
        return


    def test_CompendiumTableInterpolateDisks(self):
        # Testing of CompendiumTable.buildInterpolate for disks instance
        self.COMP.resetInterpolators()
        self.COMP.buildDiskInterpolators()
        galaxyInterpolants = np.transpose(np.stack((self.wavelength,self.inclination,
                                                    self.opticalDepth)))
        galaxyExtrapolants = np.transpose(np.stack((self.wavelength,self.inclination)))
        mask = self.opticalDepth>0.0
        atten = self.COMP.interpolate(galaxyInterpolants,galaxyExtrapolants,
                                      self.opticalDepth,opticalDepthMask=mask)
        self.assertEqual(atten.shape,self.opticalDepth.shape)
        with self.assertRaises(RuntimeError):
            self.COMP.interpolate(galaxyExtrapolants,galaxyInterpolants,
                                  self.opticalDepth,opticalDepthMask=mask)
        mask = np.zeros(len(self.opticalDepth),dtype=bool)
        atten = self.COMP.interpolate(galaxyInterpolants,galaxyExtrapolants,
                                      self.opticalDepth,opticalDepthMask=mask)
        self.assertTrue(all(atten==1.0))
        return

    def test_CompendiumTableInterpolateSpheroids(self):
        # Testing of CompendiumTable.buildInterpolate for spheroid instance   
        self.COMP.resetInterpolators()
        self.COMP.buildSpheroidInterpolators()
        galaxyInterpolants = np.transpose(np.stack((self.wavelength,self.inclination,
                                                    self.opticalDepth,self.spheroidRadius)))
        galaxyExtrapolants = np.transpose(np.stack((self.wavelength,self.inclination,
                                                    self.spheroidRadius)))
        mask = self.opticalDepth>0.0
        atten = self.COMP.interpolate(galaxyInterpolants,galaxyExtrapolants,
                                      self.opticalDepth,opticalDepthMask=mask)
        self.assertEqual(atten.shape,self.opticalDepth.shape)
        with self.assertRaises(RuntimeError):
            self.COMP.interpolate(galaxyExtrapolants,galaxyInterpolants,
                                  self.opticalDepth,opticalDepthMask=mask)
        return

    def test_CompendiumTableGetDiskAttenuation(self):
        # Testing of CompendiumTable.getDiskAttenuation
        mask = self.opticalDepth > 0.0
        atten = self.COMP.getDiskAttenuation(self.wavelength,self.inclination,self.opticalDepth,
                                             opticalDepthMask=mask)
        self.assertEqual(atten.shape,self.opticalDepth.shape)
        mask = np.zeros(len(self.opticalDepth),dtype=bool)
        atten = self.COMP.getDiskAttenuation(self.wavelength,self.inclination,self.opticalDepth,
                                             opticalDepthMask=mask)
        self.assertTrue(all(atten==1.0))
        with self.assertRaises(TypeError):
            self.COMP.getDiskAttenuation(self.wavelength,self.inclination,opticalDepthMask=mask)
        return

    def test_CompendiumTableGetSpheroidAttenuation(self):
        # Testing of CompendiumTable.getSpheroidAttenuation
        mask = self.opticalDepth > 0.0
        atten = self.COMP.getSpheroidAttenuation(self.wavelength,self.inclination,self.spheroidRadius,
                                                 self.opticalDepth,opticalDepthMask=mask)
        self.assertEqual(atten.shape,self.opticalDepth.shape)
        mask = np.zeros(len(self.opticalDepth),dtype=bool)
        atten = self.COMP.getSpheroidAttenuation(self.wavelength,self.inclination,self.spheroidRadius,
                                                 self.opticalDepth,opticalDepthMask=mask)
        self.assertTrue(all(atten==1.0))        
        with self.assertRaises(TypeError):
            self.COMP.getSpheroidAttenuation(self.wavelength,self.inclination,opticalDepthMask=mask)
        return


if __name__ == "__main__":
    unittest.main()

