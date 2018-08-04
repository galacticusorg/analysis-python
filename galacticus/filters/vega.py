#! /usr/bin/env python

import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romb
from .. import rcParams
from ..fileFormats.xmlTree import xmlTree
from ..data import GalacticusData
from . import Filter


class Spectrum(object):

    def __init__(self):        
        self.description = None
        self.origin = None
        self.units = {}
        self.spectrum = None
        return

    def reset(self):
        self.description = None
        self.origin = None
        self.units = {}
        self.spectrum = None
        return
        
    def loadFromFile(self,spectrumFile):
        TREE = xmlTree(file=spectrumFile)
        if TREE.elementExists("/spectrum/description"):
            self.description = TREE.getElement("/spectrum/description").text
        if TREE.elementExists("/spectrum/origin"):
            self.description = TREE.getElement("/spectrum/origin").text
        for unit in TREE.tree.getroot().findall("units"):
            if unit.text.starswith("wavelengths"):
                self.units["wavelengths"] = unit.text
            if unit.text.starswith("fluxes"):
                self.units["flux"] = unit.text
        DATA = TREE.tree.getroot().findall("datum")
        dtype=[("wavelength",float),("flux",float)]
        self.spectrum = np.zeros(len(DATA),dtype=dtype).view(np.recarray)
        for i,datum in enumerate(DATA):
            self.spectrum["wavelength"][i] = float(datum.text.split()[0])
            self.spectrum["flux"][i] = float(datum.text.split()[1])
        del TREE
        return
    

class Vega(Spectrum):
    """
    Vega: Class to read and store information for Vega.
    
    USAGE: VEGA = Vega([verbose=False])

         INPUT 
            verbose -- Print additional information. [Default=False] 


         Functions:  
              __call__(): Returns AB-Vega offset for specified filter transmission.
              computeFluxes(): Compute the AB and Vega fluxes for a specified 
                               filter transmission cuve.

    """
    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        super(VegaSpectrum,self).__init__()
        self.verbose = verbose
        # Identify spectrum file
        spectrumFile = rcParams.get("filters","vegaSpectrumFile",fallback=None)
        if spectrumFile is None:
            DATA = GalacticusData(verbose=self.verbose)
            spectrumFile = DATA.search("A0V_Castelli.xml")
            rcParams.set("filters","vegaSpectrumFile",spectrumFile)
        # Load spectrum
        self.loadFromFile(spectrumFile)
        # Load V-band filter
        filterFile = rcParams.get("filters","vBandFilterFile",fallback=None)
        if filterFile is None:
            DATA = GalacticusData(verbose=self.verbose)
            filterFile = DATA.search("*/Buser_V.xml")
            rcParams.set("filters","vBandFilterFile",filterFile)        
        self.VBAND = Filter()
        self.VBAND.loadFromFile(filterFile)
        self.vFluxAB = None
        self.vFluxVega = None
        return

    def __call__(self,wavelength,transmission):
        return self.abVegaOffset(wavelength,transmission)

    def computeFluxes(self,filterWavelength,filterTransmission)
        """
        computeFluxes: Compute the AB and Vega fluxes for specified tranmission curve.
        
        USAGE: fluxAB,fluxVega = Vega().computeFluxes(wavelength,transmission)
        
            INPUT
                 wavelength   -- Wavelengths for filter transmission curve.
                 transmission -- Transmission for filter transmission curve.

            OUTPUT
                 fluxAB   -- AB flux for this filter transmission.
                 fluxVega -- Vega flux for this filter transmission.

        """
        # Interpolate spectrum and transmission data
        TRANSMISSION = interp1d(filterWavelength,filterTransmission)
        kRomberg = 8
        wavelength = np.linspace(filterWavelength.min(),filterWavelength.max(),2**kRomberg+1)
        deltaWavelength = wavelength[1] - wavelength[0]
        FLUX = interp1d(self.spectrum.wavelength,self.spectrum.flux)
        # Get AB spectrum
        spectrumAB = 1.0/wavelength**2
        # Get the filtered spectrum
        filteredSpectrum = TRANSMISSION(wavelength)*FLUX(wavelength)
        filteredSpectrumAB = TRANSMISSION(wavelength)*spectrumAB
        # Compute the integrated flux.
        fluxVega = romb(filteredSpectrum,dx=deltaWavelength)
        fluxAB = romb(filteredSpectrumAB,dx=deltaWavelength)
        return fluxAB,fluxVega


    def abVegaOffset(self,wavelength,transmission):
        """
        abVegaOffset(): Compute AB-Vega offset for specified filter transmission curve.

        USAGE: offset = Vega().abVegaOffset(wavelength,transmission)
        
              INPUT
                 wavelength   -- Wavelengths for filter transmission curve.
                 transmission -- Transmission for filter transmission curve.
        
              OUTPUT
                 offset       -- AB-Vega offset for filter transmission curve.

        """
        if self.vFluxAB is None or self.vFluxVega is None:
            self.vFluxAB,self.vFluxVega = self.computeFluxes(self.VBAND.transmission.wavelength,
                                                             self.VBAND.transmission.transmission)
        # Compute fluxes for specified filter
        fluxAB,fluxVega = self.computeFluxes(wavelength,transmission)
        # Return Vega offset
        offset = 2.5*np.log10(fluxVega*self.vFluxAB/self.vFluxVega/fluxAB)
        return offset            


