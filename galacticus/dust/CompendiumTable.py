#! /usr/bin/env python

import sys,os
import numpy as np
import copy
import unittest
from scipy.interpolate import RegularGridInterpolator
from .. import rcParams
from ..data import GalacticusData
from ..fileFormats.hdf5 import HDF5

class CompendiumTable(object):
    
    def __init__(self):    
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Locate compendium attenuations
        DATA = GalacticusData()
        compendiumFile = rcParams.get("dustCompendium","attenuationsFile",\
                                          fallback="compendiumAttenuations.hdf5")
        self.file = DATA.search(compendiumFile)        
        # Determine whether to extrapolate optical depths
        self.extrapolateOpticalDepth = rcParams.getboolean(\
            "dustCompendium","extrapolateOpticalDepth",fallback=True)
        # Variables to store tables loaded from HDF5 file
        self.wavelengthTable = None
        self.inclinationTable = None
        self.opticalDepthTable = None
        self.spheroidScaleRadialTable = None
        self.attenuationDiskTable = None
        self.attenuationSpheroidTable = None
        self.extrapolationDiskTable = None
        self.extrapolationSpheroidTable = None
        self.opacity = None
        # Variables to store interpolator/extrapolator objects
        self.interpolator = None
        self.extrapolator0 = None
        self.extrapolator1 = None
        return

    def resetInterpolators(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.interpolator = None
        self.extrapolator0 = None
        self.extrapolator1 = None
        return

    def tablesLoaded(self):
        notloaded = any([self.wavelengthTable is None,self.inclinationTable is None,\
                             self.opticalDepthTable is None,self.spheroidScaleRadialTable is None,\
                             self.attenuationDiskTable is None,self.attenuationSpheroidTable is None,\
                             self.extrapolationDiskTable is None,self.extrapolationSpheroidTable is None])
        return np.invert(notloaded)
        
    
    def loadOpacity(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        FILE = HDF5(self.file,'r')
        self.opacity = copy.copy(FILE.readAttributes("/",required=["opacity"])['opacity'])
        FILE.close()
        return

    def load(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        FILE = HDF5(self.file,'r')
        self.wavelengthTable            = np.copy(FILE.readDataset('/wavelength'                       ))
        self.inclinationTable           = np.copy(FILE.readDataset('/inclination'                      ))
        self.opticalDepthTable          = np.copy(FILE.readDataset('/opticalDepth'                     ))
        self.spheroidScaleRadialTable   = np.copy(FILE.readDataset('/spheroidScaleRadial'              ))
        self.attenuationDiskTable       = np.copy(FILE.readDataset('/attenuationDisk'                  ))
        self.attenuationSpheroidTable   = np.copy(FILE.readDataset('/attenuationSpheroid'              ))
        self.extrapolationDiskTable     = np.copy(FILE.readDataset('/extrapolationCoefficientsDisk'    ))
        self.extrapolationSpheroidTable = np.copy(FILE.readDataset('/extrapolationCoefficientsSpheroid'))
        self.opacity = copy.copy(FILE.readAttributes("/",required=["opacity"])['opacity'])
        FILE.close()
        return

    def getInterpolationMask(self,opticalDepth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.extrapolateOpticalDepth:
            interpolated = opticalDepth <= self.opticalDepthTable[-1]
        else:
            interpolated = np.ones(opticalDepth.shape,dtype=bool)
        return interpolated

    def getExtrapolationMask(self,opticalDepth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.extrapolateOpticalDepth:
            extrapolated = opticalDepth >  self.opticalDepthTable[-1]
        else:
            extrapolated = np.zeros(opticalDepth.shape,dtype=bool)
        return extrapolated

    def buildDiskInterpolators(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.tablesLoaded():
            self.load()
        interpolants = (self.wavelengthTable,self.inclinationTable,self.opticalDepthTable)
        self.interpolator = RegularGridInterpolator(interpolants,self.attenuationDiskTable)
        interpolants = (self.wavelengthTable,self.inclinationTable)
        self.extrapolator0 = RegularGridInterpolator(interpolants,self.extrapolationDiskTable[0,:,:])
        self.extrapolator1 = RegularGridInterpolator(interpolants,self.extrapolationDiskTable[1,:,:])
        return

    def buildSpheroidInterpolators(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.tablesLoaded():
            self.load()
        interpolants = (self.wavelengthTable,self.inclinationTable,self.opticalDepthTable,\
                            self.spheroidScaleRadialTable)
        self.interpolator = RegularGridInterpolator(interpolants,self.attenuationSpheroidTable)
        interpolants = (self.wavelengthTable,self.inclinationTable,self.spheroidScaleRadialTable)
        self.extrapolator0 = RegularGridInterpolator(interpolants,self.extrapolationSpheroidTable[0,:,:,:])
        self.extrapolator1 = RegularGridInterpolator(interpolants,self.extrapolationSpheroidTable[1,:,:,:])
        return

    def interpolate(self,galaxyInterpolants,galaxyExtrapolants,opticalDepth,opticalDepthMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if opticalDepthMask is None:
            mask = np.ones(len(opticalDepth),dtype=bool)
        else:
            mask = opticalDepthMask
        if self.interpolator is None:
            raise RuntimeError(funcname+"(): Interpolators have not been set!")
        if len(self.interpolator.grid)!=galaxyInterpolants.shape[1]:
            raise RuntimeError(funcname+"(): Number of interpolatants incorrect. Should be: "+
                               "3 (disks) or 4 (spheroids).")
        if len(self.extrapolator0.grid)!=galaxyExtrapolants.shape[1]:
            raise RuntimeError(funcname+"(): Number of extrapolatants incorrect. Should be: "+
                               "2 (disks) or 3 (spheroids).")
        if len(self.extrapolator1.grid)!=galaxyExtrapolants.shape[1]:
            raise RuntimeError(funcname+"(): Number of extrapolatants incorrect. Should be: "+
                               "2 (disks) or 3 (spheroids).")
        interpolated = self.getInterpolationMask(opticalDepth)
        extrapolated = self.getExtrapolationMask(opticalDepth)
        # Perform the interpolation.
        attenuations = np.ones_like(opticalDepth)
        interpolateMask = np.logical_and(mask,interpolated)
        attenuations[interpolateMask] = self.interpolator(galaxyInterpolants[interpolateMask])
        # Perform the extrapolations.
        extrapolateMask = np.logical_and(mask,extrapolated)
        attenuations[extrapolateMask] = self.extrapolator0(galaxyExtrapolants[extrapolateMask])
        attenuations[extrapolateMask] += self.extrapolator1(galaxyExtrapolants[extrapolateMask])*np.log(opticalDepth[extrapolateMask])
        attenuations[extrapolateMask] = np.exp(attenuations[extrapolateMask])
        return attenuations
    
    def getDiskAttenuation(self,wavelength,inclination,opticalDepth,opticalDepthMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.resetInterpolators()
        self.buildDiskInterpolators()
        galaxyInterpolants = np.transpose(np.stack((wavelength,inclination,opticalDepth)))
        galaxyExtrapolants = np.transpose(np.stack((wavelength,inclination             )))
        attenuations = self.interpolate(galaxyInterpolants,galaxyExtrapolants,opticalDepth,\
                                            opticalDepthMask=opticalDepthMask)
        return attenuations

    def getSpheroidAttenuation(self,wavelength,inclination,spheroidScaleRadial,opticalDepth,opticalDepthMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.resetInterpolators()
        self.buildSpheroidInterpolators()
        galaxyInterpolants = np.transpose(np.stack((wavelength,inclination,opticalDepth,spheroidScaleRadial)))
        galaxyExtrapolants = np.transpose(np.stack((wavelength,inclination             ,spheroidScaleRadial)))
        attenuations = self.interpolate(galaxyInterpolants,galaxyExtrapolants,opticalDepth,\
                                            opticalDepthMask=opticalDepthMask)
        return attenuations



class UnitTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # Intitialize class
        self.COMP = CompendiumTable()
        self.COMP.load()
        # Generate some random data
        N = 100
        self.inclination = np.random.rand(N)*90.0
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

    def testLoad(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.load() function")
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
        self.assertIsNotNone(self.COMP.wavelengthTable)
        self.assertEqual(np.ndim(self.COMP.wavelengthTable),1)
        self.assertGreater(len(self.COMP.wavelengthTable),0)
        self.assertIsNotNone(self.COMP.inclinationTable)
        self.assertEqual(np.ndim(self.COMP.inclinationTable),1)
        self.assertGreater(len(self.COMP.inclinationTable),0)
        self.assertIsNotNone(self.COMP.opticalDepthTable)
        self.assertEqual(np.ndim(self.COMP.opticalDepthTable),1)
        self.assertGreater(len(self.COMP.opticalDepthTable),0)
        self.assertIsNotNone(self.COMP.spheroidScaleRadialTable)
        self.assertEqual(np.ndim(self.COMP.spheroidScaleRadialTable),1)
        self.assertGreater(len(self.COMP.spheroidScaleRadialTable),0)
        self.assertIsNotNone(self.COMP.attenuationDiskTable)
        shape = (len(self.COMP.wavelengthTable),\
                     len(self.COMP.inclinationTable),\
                     len(self.COMP.opticalDepthTable))
        self.assertEqual(self.COMP.attenuationDiskTable.shape,shape)
        self.assertIsNotNone(self.COMP.attenuationSpheroidTable)
        shape = (len(self.COMP.wavelengthTable),\
                     len(self.COMP.inclinationTable),\
                     len(self.COMP.opticalDepthTable),\
                     len(self.COMP.spheroidScaleRadialTable))
        self.assertEqual(self.COMP.attenuationSpheroidTable.shape,shape)
        self.assertIsNotNone(self.COMP.extrapolationDiskTable)
        shape = (2,len(self.COMP.wavelengthTable),\
                     len(self.COMP.inclinationTable))
        self.assertEqual(self.COMP.extrapolationDiskTable.shape,shape)
        self.assertIsNotNone(self.COMP.extrapolationSpheroidTable)
        shape = (2,len(self.COMP.wavelengthTable),\
                     len(self.COMP.inclinationTable),\
                     len(self.COMP.spheroidScaleRadialTable))
        self.assertEqual(self.COMP.extrapolationSpheroidTable.shape,shape)
        self.assertEqual(type(self.COMP.opacity),np.float64)
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetInterpolationMask(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.getInterpolationMask() function")
        opticalDepth = np.copy(self.opticalDepth)        
        self.COMP.extrapolateOpticalDepth = False
        self.assertTrue(all(self.COMP.getInterpolationMask(opticalDepth)))
        self.COMP.extrapolateOpticalDepth = True
        self.assertTrue(all(self.COMP.getInterpolationMask(opticalDepth)))
        mask = np.random.rand(len(opticalDepth))<0.1
        opticalDepth[mask] = self.COMP.opticalDepthTable[-1]*100.0
        MASK = self.COMP.getInterpolationMask(opticalDepth)
        [self.assertEqual(m,M) for m,M in zip(np.invert(mask),MASK)]
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetExrapolationMask(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.getExtrapolatoinMask() function")
        opticalDepth = np.copy(self.opticalDepth)        
        self.COMP.extrapolateOpticalDepth = False
        [self.assertFalse(m) for m in self.COMP.getExtrapolationMask(opticalDepth)]
        self.COMP.extrapolateOpticalDepth = True
        [self.assertFalse(m) for m in self.COMP.getExtrapolationMask(opticalDepth)]
        mask = np.random.rand(len(opticalDepth))<0.1
        opticalDepth[mask] = self.COMP.opticalDepthTable[-1]*100.0
        MASK = self.COMP.getExtrapolationMask(opticalDepth)
        [self.assertEqual(m,M) for m,M in zip(mask,MASK)]
        print("TEST COMPLETE")
        print("\n")
        return

    def testResetInterpolators(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.resetInterpolators() function")
        self.COMP.resetInterpolators()
        self.assertIsNone(self.COMP.interpolator)
        self.assertIsNone(self.COMP.extrapolator0)
        self.assertIsNone(self.COMP.extrapolator1)
        print("TEST COMPLETE")
        print("\n")
        return
        
    def testBuildDiskInterpolators(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.buildDiskInterpolators() function")
        self.COMP.resetInterpolators()
        self.COMP.buildDiskInterpolators()
        self.assertIsInstance(self.COMP.interpolator,RegularGridInterpolator)
        self.assertIsInstance(self.COMP.extrapolator0,RegularGridInterpolator)
        self.assertIsInstance(self.COMP.extrapolator1,RegularGridInterpolator)
        self.assertEqual(len(self.COMP.interpolator.grid),3)
        self.assertEqual(len(self.COMP.extrapolator0.grid),2)
        self.assertEqual(len(self.COMP.extrapolator1.grid),2)
        print("TEST COMPLETE")
        print("\n")
        return

    def testBuildSpheroidInterpolators(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.buildSpheroidInterpolators() function")
        self.COMP.resetInterpolators()
        self.COMP.buildSpheroidInterpolators()
        self.assertIsInstance(self.COMP.interpolator,RegularGridInterpolator)
        self.assertIsInstance(self.COMP.extrapolator0,RegularGridInterpolator)
        self.assertIsInstance(self.COMP.extrapolator1,RegularGridInterpolator)
        self.assertEqual(len(self.COMP.interpolator.grid),4)
        self.assertEqual(len(self.COMP.extrapolator0.grid),3)
        self.assertEqual(len(self.COMP.extrapolator1.grid),3)
        print("TEST COMPLETE")
        print("\n")
        return
    
    def testInterpolate(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.interpolate() function")
        print("i) Testing disk interpolation")
        self.COMP.resetInterpolators()
        self.COMP.buildDiskInterpolators()
        galaxyInterpolants = np.transpose(np.stack((self.wavelength,self.inclination,\
                                                        self.opticalDepth)))
        galaxyExtrapolants = np.transpose(np.stack((self.wavelength,self.inclination)))
        mask = self.opticalDepth>0.0
        atten = self.COMP.interpolate(galaxyInterpolants,galaxyExtrapolants,\
                                          self.opticalDepth,opticalDepthMask=mask)
        self.assertEqual(atten.shape,self.opticalDepth.shape)
        self.assertRaises(RuntimeError, self.COMP.interpolate,galaxyExtrapolants,galaxyInterpolants,\
                              self.opticalDepth,opticalDepthMask=mask)
        mask = np.zeros(len(self.opticalDepth),dtype=bool)
        atten = self.COMP.interpolate(galaxyInterpolants,galaxyExtrapolants,\
                                          self.opticalDepth,opticalDepthMask=mask)
        self.assertTrue(all(atten==1.0))
        print("ii) Testing spheroid interpolation")
        self.COMP.resetInterpolators()
        self.COMP.buildSpheroidInterpolators()
        galaxyInterpolants = np.transpose(np.stack((self.wavelength,self.inclination,\
                                                        self.opticalDepth,self.spheroidRadius)))
        galaxyExtrapolants = np.transpose(np.stack((self.wavelength,self.inclination,\
                                                        self.spheroidRadius)))
        mask = self.opticalDepth>0.0
        atten = self.COMP.interpolate(galaxyInterpolants,galaxyExtrapolants,\
                                          self.opticalDepth,opticalDepthMask=mask)
        self.assertEqual(atten.shape,self.opticalDepth.shape)
        self.assertRaises(RuntimeError, self.COMP.interpolate,galaxyExtrapolants,galaxyInterpolants,\
                              self.opticalDepth,opticalDepthMask=mask)
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetDiskAttenuation(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.getDiskAttenuation() function")        
        mask = self.opticalDepth > 0.0
        atten = self.COMP.getDiskAttenuation(self.wavelength,self.inclination,self.opticalDepth,\
                                                 opticalDepthMask=mask)
        self.assertEqual(atten.shape,self.opticalDepth.shape)
        mask = np.zeros(len(self.opticalDepth),dtype=bool)        
        atten = self.COMP.getDiskAttenuation(self.wavelength,self.inclination,self.opticalDepth,\
                                                 opticalDepthMask=mask)
        self.assertTrue(all(atten==1.0))
        self.assertRaises(TypeError,self.COMP.getDiskAttenuation,self.wavelength,self.inclination,\
                              opticalDepthMask=mask)
        print("TEST COMPLETE")
        print("\n")
        return
        
    def testGetSpheroidAttenuation(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Compendium Attenuation Table: "+funcname)
        print("Testing CompendiumTable.getSpheroidAttenuation() function")        
        mask = self.opticalDepth > 0.0
        atten = self.COMP.getSpheroidAttenuation(self.wavelength,self.inclination,self.spheroidRadius,\
                                                     self.opticalDepth,opticalDepthMask=mask)
        self.assertEqual(atten.shape,self.opticalDepth.shape)
        mask = np.zeros(len(self.opticalDepth),dtype=bool)        
        atten = self.COMP.getSpheroidAttenuation(self.wavelength,self.inclination,self.spheroidRadius,\
                                                 self.opticalDepth,opticalDepthMask=mask)
        self.assertTrue(all(atten==1.0))
        self.assertRaises(TypeError,self.COMP.getSpheroidAttenuation,self.wavelength,self.inclination,\
                              opticalDepthMask=mask)
        print("TEST COMPLETE")
        print("\n")
        return
        
