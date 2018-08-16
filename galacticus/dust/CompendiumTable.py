#! /usr/bin/env python

import sys,os
import nunmpy as np
import copy
from scipy.interpolate import RegularGridInterpolator
from .. import rcParams
from ..data import GalacticusData
from ..fileFormats.hdf5 import HDF5

class CompendiumTable(object):
    
    def __init__(self):    
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
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

    def load(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        DATA = GalacticusData()
        compendiumFile = rcParams.get("dustCompendium","attenuationsFile",\
                                          fallback="compendiumAttenuations.hdf5"))
        attenuationsFile = DATA.search(compendiumFile)        
        FILE = HDF5(attenuationsFile,'r')
        self.wavelengthTable            = np.copy(FILE.getDataset('/wavelength'                       ))
        self.inclinationTable           = np.copy(FILE.getDataset('/inclination'                      ))
        self.opticalDepthTable          = np.copy(FILE.getDataset('/opticalDepth'                     ))
        self.spheroidScaleRadialTable   = np.copy(FILE.getDataset('/spheroidScaleRadial'              ))
        self.attenuationDiskTable       = np.copy(FILE.getDataset('/attenuationDisk'                  ))
        self.attenuationSpheroidTable   = np.copy(FILE.getDataset('/attenuationSpheroid'              ))
        self.extrapolationDiskTable     = np.copy(FILE.getDataset('/extrapolationCoefficientsDisk'    ))
        self.extrapolationSpheroidTable = np.copy(FILE.getDataset('/extrapolationCoefficientsSpheroid'))
        self.opacity                    = copy.copy(FILE.readAttributes("/",required=["opacity"])['opacity'])
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
        if self.opticalDepth is None:
            self.load()
        interpolants = (self.wavelengthTable,self.inclinationTable,self.opticalDepthTable)
        self.interpolator = RegularGridInterpolator(interpolants,self.attenuationDiskTable)
        interpolants = (self.wavelengthTable,self.inclinationTable)
        self.extrapolator0 = RegularGridInterpolator(interpolants,self.extrapolationDiskTable[0,:,:])
        self.extrapolator1 = RegularGridInterpolator(interpolants,self.extrapolationDiskTable[1,:,:])
        return

    def buildSpheroidInterpolators(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.opticalDepth is None:
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
        assert(len(self.interpolator.grid)==galaxyInterpolants.shape[1])
        assert(len(self.extrapolator0.grid)==galaxyExtrapolants.shape[1])
        assert(len(self.extrapolator1.grid)==galaxyExtrapolants.shape[1])
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

