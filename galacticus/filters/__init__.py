#! /usr/bin/env python

import __future__
import sys
import numpy as np

def computeEffectiveWavelength(wavelength,transmission):
    """
    computeEffectiveWavelength: Compute the effective wavelength for a filter transmission.

    USAGE: effectiveWavelength = computeEffectiveWavelength(wavelength,transmission)

       INPUT 
            wavelength   -- Numpy array of wavelengths.
            transmission -- Numpy array of filter transmssion curve.

       OUTPUT
            effectiveWavelength -- Effactive wavelength for transmission curve.
    """
    return np.sum(wavelength*transmission)/np.sum(transmission)


class Filter(object):

    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.file = None
        self.transmission = None
        self.description = None
        self.effectiveWavelength = None
        self.vegaOffset = None
        self.name = None
        self.origin = None
        self.url = None
        return

    def reset(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.file = None
        self.transmission = None
        self.description = None
        self.effectiveWavelength = None
        self.vegaOffset = None
        self.name = None
        self.origin = None
        self.url = None
        return

    def setEffectiveWavelength(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.transmission is None:
            raise ValueError(funcname+"(): no filter transmission has been set.")
        self.effectiveWavelength = computeEffectiveWavelength(self.transmission.wavelength,\
                                                                  self.transmission.transmission)
        return

    def setTransmission(self,wavelength,response):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store transmission
        if len(wavelength) != len(response):
            raise ValueError(funcname+"(): wavelength and response arrays are different length.")
        dtype = [("wavelength",float),("transmission",float)]
        self.transmission = np.zeros(len(wavelength),dtype=dtype).view(np.recarray)
        self.transmission.wavelength = wavelength
        self.transmission.transmission = response
        self.setEffectiveWavelength()
        return


