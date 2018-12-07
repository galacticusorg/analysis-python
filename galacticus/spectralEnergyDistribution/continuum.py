#! /usr/bin/env python

import numpy as np
import fnmatch
from .. import rcParams
from ..io import GalacticusHDF5
from ..filters.tophats import TopHat
from ..constants import luminosityAB
from ..constants import angstrom,speedOfLight
from ..constants import plancksConstant
from . import getSpectralEnergyDistributionWavelengths,parseDatasetName

class sedContinuum(object):
    
    def __init__(self,galaxies):
        self.galaxies = galaxies
        return

    def identifyTopHatLuminosityDatasets(self,redshift,MATCH):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dsets = self.galaxies.GH5Obj.availableDatasets(redshift)
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
        topHats = fnmatch.filter(dsets,search)
        # If 'total', replace compoent
        if fnmatch.fnmatch(MATCH.group("component"),"total"):
            topHats = [Filter.replace("disk","total") for Filter in topHats]
        return topHats

    @classmethod
    def selectWavelengthRange(cls,topHats,MATCH):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Sort top hats by wavelength
        wavelengths = cls.extractTopHatWavelengths(topHats,sortTopHats=True)
        # Get wavelength limits of 
        lowerWavelength = float(MATCH.group("lowerWavelength"))
        upperWavelength = float(MATCH.group("upperWavelength"))
        # Check wavelength limits
        wavelengthRangeStr = str(wavelengths.min())+"A-"+str(wavelengths.max())+"A"
        if lowerWavelength < wavelengths.min():
            msg = funcname+"(): lower wavelength limit of "+str(lowerWavelength)+\
                  "A is outside wavelength range of top hat filters: "+wavelengthRangeStr+"."
            raise ValueError(msg)
        if upperWavelength > wavelengths.max():
            msg = funcname+"(): upper wavelength limit of "+str(lowerWavelength)+\
                  "A is outside wavelength range of top hat filters: "+wavelengthRangeStr+"."
            raise ValueError(msg)
        # Build mask to remove top hat filters outside wavelength range. Allow for one top hat filter
        # outside minimum and maximum for purposes of interpolation)
        mask = np.logical_and(wavelengths>=lowerWavelength,wavelengths<=upperWavelength)
        ipass = [i for i, m in enumerate(mask) if m]
        ilow = ipass[0]
        if ilow > 0:
            ilow -= 1            
        mask[ilow] = True
        iupp = ipass[-1]
        if iupp < len(mask)-1:
            iupp += 1            
        mask[iupp] = True
        # Apply mask
        wavelengths = wavelengths[mask]
        topHats = list(np.array(topHats)[mask])
        return wavelengths

    def getContinuumLuminosities(self,redshift,MATCH):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get list of top hat luminosities to extract
        topHats = self.identifyTopHatLuminosityDatasets(redshift,MATCH)
        # Select top hats inside specified wavelength range
        wavelengths = self.selectWavelengthRange(topHats,MATCH)
        # Read top hat luminosities
        TOPHATS = self.galaxies.get(redshift,properties=[topHats])    
        # Create 2D grid of luminosities
        luminosities = np.stack([TOPHATS[dset].data for dset in topHats],axis=1)
        return luminosities,wavelengths

    @classmethod
    def interpolateContinuum(cls,wavelengths,luminosities,newWavelengths):
        funcname = sys._getframe().f_code.co_name
        method = rcParams.get("spectralEnergyDistribution","continuumInterpolationMethod",
                              fallback="linear")
        f = interp1d(wavelengths,luminosities,kind=method,axis=1)
        return f(newWavelengths)

    @classmethod
    def addContinuumNoise(cls,wavelengths,continuum,SNR):
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
        luminosities,topHatWavelengths = self.getContinuumLuminosities(redshift,MATCH)
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
        filterNames = [fnmatch.filter(dset.split(":"),"*adaptiveResolutionTopHat*") for dset in topHats]
        wavelengths = np.array([TopHat.getFilterSize(name)[0] for name in filterNames])
        isort = np.argsort(wavelengths)
        wavelengths = wavelengths[isort]
        if sortTophats:
            topHats = list(np.array(topHats)[isort])
        return wavelengths

    
        




