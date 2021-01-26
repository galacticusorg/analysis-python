#! /usr/bin/env python                                                                                                                                                          
import sys
import numpy as np
from scipy.special import erf
from ..constants import Pi

class LineProfiles(object):

    @classmethod
    def gaussian(cls,wavelengths,lineWavelength,lineLuminosity,FWHM):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute sigma for Gaussian
        sigma = FWHM/(2.0*np.sqrt(2.0*np.log(2.0)))
        sigma =  np.stack([sigma]*len(wavelengths),axis=1).reshape(len(sigma),-1)
        # Compute amplitude for Gaussian
        amplitude = np.stack([lineLuminosity]*len(wavelengths),axis=1).reshape(len(lineLuminosity),-1)
        # Compute luminosity
        wavelengthsDelta = np.gradient(wavelengths)
        wavelengthsGrid = np.concatenate([wavelengths]*len(lineLuminosity)).reshape(-1,len(wavelengths))
        wavelengthsDeltaGrid = np.concatenate([wavelengthsDelta]*len(lineLuminosity)).reshape(-1,len(wavelengthsDelta))
        lineWavelengths = np.stack([lineWavelength]*len(wavelengths),axis=1).reshape(len(lineLuminosity),-1)
        luminosity = amplitude*0.5*(erf((wavelengthsGrid+0.5*wavelengthsDeltaGrid-lineWavelengths)/np.sqrt(2.0)/sigma)-erf((wavelengthsGrid-0.5*wavelengthsDeltaGrid-lineWavelengths)/np.sqrt(2.0)/sigma))/wavelengthsDeltaGrid
        return luminosity
