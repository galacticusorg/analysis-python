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
        compendiumFile = rcParams.get("dustCompendium","attenuationsFile",
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
        notloaded = any([self.wavelengthTable is None,self.inclinationTable is None,
                         self.opticalDepthTable is None,self.spheroidScaleRadialTable is None,
                         self.attenuationDiskTable is None,self.attenuationSpheroidTable is None,
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
        FILE.close()
        self.loadOpacity()
        return

    def assertSpheroidRadialScaleInRange(self,spheroidScaleRadial):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.tablesLoaded():
            self.load()
        if (any(spheroidScaleRadial < self.spheroidScaleRadialTable[0])):
                raise RuntimeError(funcname+"(): galaxies with spheroid radial scale < "+
                                   str(self.spheroidScaleRadialTable[0])+" present - out of range")
        if (any(spheroidScaleRadial > self.spheroidScaleRadialTable[-1])):
            raise RuntimeError(funcname+"(): galaxies with spheroid radial scale > "+
                               str(self.spheroidScaleRadialTable[-1])+" present - out of range")
        return True
    
    def assertInclinationInRange(self,inclination):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.tablesLoaded():
            self.load()
        if (any(inclination < 0.0)):
            raise RuntimeError(funcname+"(): galaxies with inclination < 0 present - this is not permitted!")
        if (any(inclination > 90.0)):
            raise RuntimeError(funcname+"(): galaxies with inclination > 90 present - this is not permitted!")
        if (any(inclination  < self.inclinationTable[0])):
            raise RuntimeError(funcname+"(): galaxies with inclination < "+str(self.inclinationTable[0])+" present - out of range!")
        if (any(inclination  > self.inclinationTable[-1])):
            raise RuntimeError(funcname+"(): galaxies with inclination > "+str(self.inclinationTable[-1])+" present - out of range!")
        return True

    def assertWavelengthInRange(self,wavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.tablesLoaded():
            self.load()            
        if (any(wavelength < self.wavelengthTable[0])):
            raise RuntimeError(funcname+"(): galaxies with wavelength < "+
                               str(self.wavelengthTable[0])+" present - out of range!")
        if (any(wavelength > self.wavelengthTable[-1])):
            raise RuntimeError(funcname+"(): galaxies with wavelength > "+
                               str(self.wavelengthTable[-1])+" present - out of range!")
        return True

    def assertOpticalDepthInRange(self,opticalDepth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.tablesLoaded():
            self.load()            
        if (any(opticalDepth < self.opticalDepthTable[0])):
            raise RuntimeError(funcname+"(): galaxies with optical depth < "+
                               str(self.opticalDepthTable[0])+" present - out of range!")                                                      
        if (any(opticalDepth > self.opticalDepthTable[-1]) and not self.extrapolateOpticalDepth):
            raise RuntimeError(funcname+"(): galaxies with optical depth > "+
                               str(self.opticalDepthTable[-1])+" present - out of range")
        return True
    
    def getInterpolationMask(self,opticalDepth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        nanMask = np.isnan(opticalDepth)
        opticalDepth[nanMask] = self.opticalDepthTable[-1]*100.0
        if self.extrapolateOpticalDepth:
            interpolated = opticalDepth <= self.opticalDepthTable[-1]
        else:
            interpolated = np.ones(opticalDepth.shape,dtype=bool)
        interpolated = np.logical_and(interpolated,np.invert(nanMask))
        return interpolated

    def getExtrapolationMask(self,opticalDepth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        nanMask = np.isnan(opticalDepth)
        opticalDepth[nanMask] = self.opticalDepthTable[-1]
        if self.extrapolateOpticalDepth:
            extrapolated = opticalDepth > self.opticalDepthTable[-1]
        else:
            extrapolated = np.zeros(opticalDepth.shape,dtype=bool)
        extrapolated = np.logical_and(extrapolated,np.invert(nanMask))
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
        interpolants = (self.wavelengthTable,self.inclinationTable,self.opticalDepthTable,
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
        self.assertWavelengthInRange(wavelength[opticalDepthMask])
        self.assertInclinationInRange(inclination[opticalDepthMask])
        self.assertOpticalDepthInRange(opticalDepth[opticalDepthMask])
        galaxyInterpolants = np.transpose(np.stack((wavelength,inclination,opticalDepth)))
        galaxyExtrapolants = np.transpose(np.stack((wavelength,inclination             )))
        attenuations = self.interpolate(galaxyInterpolants,galaxyExtrapolants,opticalDepth,
                                        opticalDepthMask=opticalDepthMask)
        return attenuations

    def getSpheroidAttenuation(self,wavelength,inclination,spheroidScaleRadial,opticalDepth,opticalDepthMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.resetInterpolators()
        self.buildSpheroidInterpolators()
        self.assertWavelengthInRange(wavelength[opticalDepthMask])
        self.assertInclinationInRange(inclination[opticalDepthMask])
        self.assertOpticalDepthInRange(opticalDepth[opticalDepthMask])
        self.assertSpheroidRadialScaleInRange(spheroidScaleRadial[opticalDepthMask])        
        galaxyInterpolants = np.transpose(np.stack((wavelength,inclination,opticalDepth,spheroidScaleRadial)))
        galaxyExtrapolants = np.transpose(np.stack((wavelength,inclination             ,spheroidScaleRadial)))
        attenuations = self.interpolate(galaxyInterpolants,galaxyExtrapolants,opticalDepth,
                                        opticalDepthMask=opticalDepthMask)
        return attenuations


