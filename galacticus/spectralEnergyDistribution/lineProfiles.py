#! /usr/bin/env python                                                                                                                                                          
import sys
import numpy as np
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
        amplitude /= (sigma*np.sqrt(2.0*Pi))
        # Compute luminosity
        wavelengthsGrid = np.concatenate([wavelengths]*len(lineLuminosity)).reshape(-1,len(wavelengths))
        lineWavelengths = np.stack([lineWavelength]*len(wavelengths),axis=1).reshape(len(lineLuminosity),-1)
        luminosity = amplitude*np.exp(-((wavelengthsGrid-lineWavelengths)**2)/(2.0*(sigma**2)))
        return luminosity
