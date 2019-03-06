#! /usr/bin/env python

import os,sys,re
import numpy as np
from . import Filter
from .vega import Vega
from ..data import GalacticusData
from ..errors import ParseError
from ..stellarPopulations import SyntheticStellarPopulation

def getTransmissionCurve(wavelengthCentral,wavelengthWidth,transmissionSize=1000,
                         edgesBufferFraction=0.1):
    fraction = 0.5 + edgesBufferFraction
    lowerLimit = wavelengthCentral - wavelengthWidth*fraction
    upperLimit = wavelengthCentral + wavelengthWidth*fraction
    dtype = [("wavelength",float),("transmission",float)]
    transmission = np.zeros(transmissionSize,dtype=dtype).view(np.recarray)
    transmission.wavelength = np.linspace(lowerLimit,upperLimit,transmissionSize)
    lowerEdge = wavelengthCentral - wavelengthWidth/2.0
    upperEdge = wavelengthCentral + wavelengthWidth/2.0
    inside = np.logical_and(transmission.wavelength>=lowerEdge,
                            transmission.wavelength<=upperEdge)
    np.place(transmission.transmission,inside,1.0)
    del inside
    return transmission


class TopHat(object):

    def __init__(self,verbose=False,VegaObj=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        self.DATA = GalacticusData(verbose=self.verbose)
        if VegaObj is None:
            self.VEGA = Vega(verbose=self.verbose)
        else:
            self.VEGA = VegaObj
        return

    @classmethod
    def getFilterSize(cls,filterName):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        if "adaptiveResolutionTopHat" in filterName:
            wavelengthCentral,wavelengthWidth = cls.getFilterSizeAdaptiveResolution(filterName)
        elif "fixedResolutionTopHat" in filterName:
            wavelengthCentral,wavelengthWidth = cls.getFilterSizeFixedResolution(filterName)
        else:
            raise ParseError(funcname+"(): Filter name not recognized. Filter name should be "+\
                                 "adaptiveResolutionTopHat_<center>_<width> or "+\
                                 "fixedResolutionTopHat_<center>_<resolution>.")
        return wavelengthCentral,wavelengthWidth

    @classmethod
    def getFilterSizeAdaptiveResolution(cls,filterName):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        search = "adaptiveResolutionTopHat_(?P<center>[\d\.]+)_(?P<width>[\d\.]+)"
        MATCH = re.search(search,filterName) 
        if MATCH is None:
            raise ParseError(funcname+"(): Unable to parse '"+filterName+"'.")
        return float(MATCH.group('center')),float(MATCH.group('width'))

    @classmethod
    def getFilterSizeFixedResolution(cls,filterName):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        search = "fixedResolutionTopHat_(?P<center>[\d\.]+)_(?P<resolution>[\d\.]+)"
        MATCH = re.search(search,filterName) 
        if MATCH is None:
            raise ParseError(funcname+"(): Unable to parse '"+filterName+"'.")
        wavelengthCentral = float(MATCH.group('center'))
        resolution = float(MATCH.group('resolution'))
        wavelengthRatio = (np.sqrt(4.0*resolution**2+1.0)+1.0)/(np.sqrt(4.0*resolution**2+1.0)-1.0)
        wavelengthMinimum = wavelengthCentral*(np.sqrt(4.0*resolution**2+1.0)-1.0)/2.0/resolution
        wavelengthMinimum /= wavelengthRatio
        wavelengthMaximum = wavelengthCentral*(np.sqrt(4.0*resolution**2+1.0)+1.0)/2.0/resolution
        wavelengthMaximum /= wavelengthRatio
        wavelengthWidth = wavelengthMaximum - wavelengthMinimum
        return wavelengthCentral,wavelengthWidth

    def create(self,filterName,writeToFile=True,transmissionSize=1000,edgesBufferFraction=0.1):
        FILTER = Filter()
        FILTER.name = filterName
        wavelengthCentral,wavelengthWidth = self.getFilterSize(filterName)
        FILTER.transmission = getTransmissionCurve(wavelengthCentral,wavelengthWidth,\
                                                       transmissionSize=transmissionSize,\
                                                       edgesBufferFraction=edgesBufferFraction)
        FILTER.setEffectiveWavelength()
        FILTER.vegaOffset = self.VEGA.abVegaOffset(FILTER.transmission.wavelength,\
                                                       FILTER.transmission.transmission)
        FILTER.origin = "Galacticus source code"
        FILTER.description = "SED top hat filter centered on "+str(wavelengthCentral)+\
            " Angstroms with width "+str(wavelengthWidth)+" Angstroms."
        FILTER.url = "None"
        if writeToFile:
            FILTER.writeToFile()
        return FILTER

    
class adaptiveResolutionTopHatArray(object):

    def __init__(self,lambdaMin,lambdaMax,lambdaWidth,z):
        self.obsMin = lambdaMin
        self.obsMax = lambdaMax
        self.obsWidth = lambdaWidth
        self.restMin = lambdaMin/(1.0+z)
        self.restMax = lambdaMax/(1.0+z)
        self.restWidth = lambdaWidth/(1.0+z)
        self.wavelengthCentral = []
        self.wavelenghtWidth = []        
        self._built = False
        return

    def reset(self):
        self.wavelengthCentral = []
        self.wavelenghtWidth = []        
        self._built = False
        return

    def insideObservedRange(self,wavelength):
        inside = wavelength>=self.obsMin and wavelength<=self.obsMax
        return inside

    def insideRestRange(self,wavelength):
        inside = wavelength>=self.restMin and wavelength<=self.restMax
        return inside
    
    def build(self,SSP):
        # Build first entry
        self.wavelengthCentral.append(self.restMin)
        width = np.maximum(self.restMin,SPS.wavelengthResolution(self.restMin))
        self.wavelengthWidth.append(width)
        # Loop to construct the array
        while self.wavelengthCentral[-1] < self.obsMax:
            lowerEdge = self.wavelengthCentral[-1] + self.wavelengthWidth[-1]/2.0
            spsWidth = SSP.wavelngthResolution(lowerEdge)
            if insideRestRange(lowerEdge):
                width = np.maximum(self.restWidth,spsWidth)
                center = lowerEdge + width/2.0
            elif insideObservedRange(wavelength):
                width = np.maximum(self.obsWidth,spsWidth)
                center = lowerEdge + width/2.0
            else:
                center = self.obsMin
                width = np.maximum(self.obsWidth,SSP.wavelngthResolution(self.obsMin))
                if center - width/2.0 < lowedEdge:
                    width = np.maximum(self.obsWidth,spsWidth)
                    center = lowerEdge + wdith/2.0                    
            self.wavelengthCentral.append(center)
            self.wavelengthWidth.append(width)
        return

    



                
            

