#! /usr/bin/env python

import sys,re
import numpy as np
import fnmatch
import warnings
from scipy.interpolate import interp1d
from scipy.stats import norm
from .. import rcParams
from ..errors import ParseError
from ..io import GalacticusHDF5
from ..filters.topHats import TopHat
from ..constants import luminosityAB
from ..constants import angstrom,speedOfLight
from ..constants import plancksConstant
from . import getSpectralEnergyDistributionWavelengths,parseDatasetName

class Continuum(object):
    
    def __init__(self,galaxies):
        self.galaxies = galaxies
        return

    def identifyTopHatLuminosityDatasets(self,redshift,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = parseDatasetName(propertyName)
        if MATCH is None:
            raise ParseError(funcname+"(): Cannot parse SED dataset name'" +propertyName+"'.")        
        # Construct search string for identifying top hat filters
        if fnmatch.fnmatch(MATCH.group("component"),"total"):
            search = "diskLuminositiesStellar"
        else:
            search = MATCH.group("component")+"LuminositiesStellar"
        search = search+":adaptiveResolutionTopHat_*_*:"
        search = search+MATCH.group("frame")
        search = search+MATCH.group("redshiftString")
        if MATCH.group("dust") is not None:
            search = search+MATCH.group("dust")
        # Search for filters
        dsets = self.galaxies.GH5Obj.availableDatasets(redshift)
        topHats = fnmatch.filter(dsets,search)
        # If 'total', replace compoent
        if fnmatch.fnmatch(MATCH.group("component"),"total"):
            topHats = [Filter.replace("disk","total") for Filter in topHats]
        return topHats

    @classmethod
    def selectWavelengthRange(cls,topHats,lowerWavelength,upperWavelength):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Sort top hats by wavelength
        wavelengths,topHats = cls.extractTopHatWavelengths(topHats,sortTopHats=True)
        # Check wavelength limits
        if lowerWavelength >= upperWavelength:
            raise ValueError(funcname+"(): Lower wavelength limit is equal to or "+
                             "greater than upper wavelength limit.")
        if lowerWavelength >=  wavelengths.max():
            raise ValueError(funcname+"(): Lower wavelength limit greater than or equal to "+
                             "upper wavelength range of SED: "+str(wavelengths.max())+" A.")
        if upperWavelength <=  wavelengths.min():
            raise ValueError(funcname+"(): Lower wavelength limit less than or equal to "+
                             "lower wavelength range of SED: "+str(wavelengths.min())+" A.")
        wavelengthRangeStr = str(wavelengths.min())+"A-"+str(wavelengths.max())+"A"        
        if lowerWavelength < wavelengths.min():
            msg = funcname+"(): lower wavelength limit of "+str(lowerWavelength)+\
                  "A is outside wavelength range of top hat filters: "+wavelengthRangeStr+"."+\
                  " Setting lower wavelength limit to lower wavelength range."
            warnings.warn(msg)
            lowerWavelength = wavelengths.min()
        if upperWavelength > wavelengths.max():
            msg = funcname+"(): upper wavelength limit of "+str(lowerWavelength)+\
                  "A is outside wavelength range of top hat filters: "+wavelengthRangeStr+"."+\
                     " Setting upper wavelength limit to upper wavelength range."
            warnings.warn(msg)
            upperWavelength = wavelengths.max()
        # Build mask to remove top hat filters outside wavelength range. Allow for one top hat filter
        # outside minimum and maximum for purposes of interpolation)
        mask = np.logical_and(wavelengths>=lowerWavelength,wavelengths<=upperWavelength)
        upp = np.argwhere(mask).max() + 1
        if upp == len(mask):
            upp = len(mask)-1
        mask[upp] = True
        low = np.argwhere(mask).min() - 1
        if low == -1:
            low += 1
        mask[low] = True
        # Apply mask
        wavelengths = wavelengths[mask]
        topHats = list(np.array(topHats)[mask])
        del mask
        return wavelengths,topHats

    def getContinuumLuminosities(self,redshift,propertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get list of top hat luminosities to extract
        topHats = self.identifyTopHatLuminosityDatasets(redshift,propertyName)
        # Select top hats inside specified wavelength range
        MATCH = parseDatasetName(propertyName)
        lowerWavelength = float(MATCH.group("lowerWavelength"))
        upperWavelength = float(MATCH.group("upperWavelength"))
        wavelengths,topHats = self.selectWavelengthRange(topHats,lowerWavelength,upperWavelength)
        # Read top hat luminosities
        TOPHATS = self.galaxies.get(redshift,properties=topHats)    
        # Create 2D grid of luminosities
        luminosities = np.stack([TOPHATS[dset].data for dset in topHats],axis=1)
        return luminosities,wavelengths

    @classmethod
    def interpolateContinuum(cls,wavelengths,luminosities,newWavelengths):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        method = rcParams.get("spectralEnergyDistribution","continuumInterpolationMethod",
                              fallback="linear")
        if luminosities.shape[1] != len(wavelengths):
            raise ValueError(funcname+"(): Luminosities array must have shape: (n,len(wavelengths)).")
        f = interp1d(wavelengths,luminosities,kind=method,axis=1,fill_value="extrapolate")
        return f(newWavelengths)

    @classmethod
    def addContinuumNoise(cls,wavelengths,continuum,SNR):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        if SNR<=0:
            raise ValueError(funcname+"(): S/N ratio must have positive, non-zero value.")
        # Get Poisson error on count rate of photons
        energy = speedOfLight*plancksConstant/np.stack([wavelengths]*continuum.shape[0])*angstrom
        counts = continuum*luminosityAB/energy
        # Perturb counts and convert back to luminosities
        counts = norm.rvs(loc=counts,scale=counts/SNR)
        continuum = np.copy(counts*energy/luminosityAB)
        return continuum        

    def get(self,propertyName,redshift):
        # Parse dataset name
        MATCH = parseDatasetName(propertyName)
        # Extract top hat luminosities
        luminosities,topHatWavelengths = self.getContinuumLuminosities(redshift,MATCH.group(0))
        # Create array of wavelengths from SED dataset name
        wavelengths = getSpectralEnergyDistributionWavelengths(MATCH.group(0))
        # Interpolate luminosities onto new wavelengths
        continuum = self.interpolateContinuum(topHatWavelengths,luminosities,wavelengths)
        # Add continuum noise if specified
        if MATCH.group('snr') is not None:
            continuum = self.addContinuumNoise(wavelengths,continuum,float(MATCH.group('snr')))
        return wavelengths,continuum
    
    @classmethod
    def extractTopHatWavelengths(cls,topHats,sortTopHats=True):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get list of filter names from list of top hat dataset names
        filterNames = [fnmatch.filter(dset.split(":"),"*adaptiveResolutionTopHat*")[0] for dset in topHats]
        # Extract list of wavelengths
        wavelengths = np.array([TopHat.getFilterSize(name)[0] for name in filterNames])
        # Sort wavelengths into ascending order
        isort = np.argsort(wavelengths)
        wavelengths = wavelengths[isort]
        # Sort top hat names if specified
        if sortTopHats:
            topHats = list(np.array(topHats)[isort])
        return wavelengths,topHats

    def getAvailableWavelengthRange(self,redshift,propertyName):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        topHats = self.identifyTopHatLuminosityDatasets(redshift,propertyName)
        wavelengths,topHats = self.extractTopHatWavelengths(topHats)
        return wavelengths
    
        




