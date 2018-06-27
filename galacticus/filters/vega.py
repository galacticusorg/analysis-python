#! /usr/bin/env python

import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romb
import xml.etree.ElementTree as ET
from ..data import GalacticusData
from .io import loadFilterFromFile
from . import Filter

class VegaSpectrum(object):
    """
    VegaSpectrum: Class to read and store the spectrim for Vega.
    
    USAGE: VEGA = VegaSpectrum([path=None],[fileName=A0V_Castelli.xml],[verbose=False])

         INPUT 
               path -- Path to datasets repository. If None, will search for path in 
                       environment variables (stored as 'GALACTICUS_DATASETS'). 
                       [Default=None] 
           fileName -- Name of Vega spectrum file. [Default=A0V_Castelli.xml]
            verbose -- Print additional information. [Default=False] 


         Functions:  
              __call__ : Returns 'spectrum' class attribute, which is a numpy record array
                         with records 'wavelength' and 'flux'.
              computeFluxes: Compute the AB and Vega fluxes for a specified filter 
                             transmission cuve.

    """
    def __init__(self,path=None,fileName="A0V_Castelli.xml",verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        # Identify spectrum file
        DATA = GalacticusData(path=path,verbose=self.verbose)
        if not fileName.endswith(".xml"):
            fileName = fileName + ".xml"
        spectrumFile = DATA.search(fileName)
        # Open file
        xmlStruct = ET.parse(spectrumFile)
        xmlRoot = xmlStruct.getroot()
        xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p}
        data = xmlRoot.findall("datum")
        # Load spectrum
        self.spectrum = np.zeros(len(data),dtype=[("wavelength",float),("flux",float)])
        for i,datum in enumerate(data):
            self.spectrum["wavelength"][i] = float(datum.text.split()[0])
            self.spectrum["flux"][i] = float(datum.text.split()[1])
        self.spectrum = self.spectrum.view(np.recarray)
        isort = np.argsort(self.spectrum["wavelength"])
        self.spectrum["wavelength"] = self.spectrum["wavelength"][isort]
        self.spectrum["flux"] = self.spectrum["flux"][isort]
        # Load additional information
        self.description = xmlRoot.find("description").text
        self.origin = xmlRoot.find("origin").text
        return

    def computeFluxes(self,filterWavelength,filterTransmission,kRomberg=8,**kwargs):
        """
        computeFluxes: Compute the AB and Vega fluxes for specified tranmission curve.
        
        USAGE: fluxAB,fluxVega = VegaSpectrum().computeFluxes(wavelength,transmission,[kRomberg=8],**kwargs)
        
            INPUT
                 wavelength   -- Wavelengths for filter transmission curve.
                 transmission -- Transmission for filter transmission curve.
                 kRomberg     -- Number of k-nodes for Romberg integration. [Default=8]
                 **kwrgs      -- Keywords arguments to pass to scipy.interpolate.interp1d.

            OUTPUT
                 fluxAB   -- AB flux for this filter transmission.
                 fluxVega -- Vega flux for this filter transmission.

        """
        # Interpolate spectrum and transmission data
        TRANSMISSION = interp1d(filterWavelength,filterTransmission,**kwargs)
        wavelength = np.linspace(filterWavelength.min(),filterWavelength.max(),2**kRomberg+1)
        deltaWavelength = wavelength[1] - wavelength[0]
        FLUX = interp1d(self.spectrum.wavelength,self.spectrum.flux,**kwargs)
        # Get AB spectrum
        spectrumAB = 1.0/wavelength**2
        # Get the filtered spectrum
        filteredSpectrum = TRANSMISSION(wavelength)*FLUX(wavelength)
        filteredSpectrumAB = TRANSMISSION(wavelength)*spectrumAB
        # Compute the integrated flux.
        fluxVega = romb(filteredSpectrum,dx=deltaWavelength)
        fluxAB = romb(filteredSpectrumAB,dx=deltaWavelength)
        return fluxAB,fluxVega
            
    def __call__(self):
        return self.spectrum

    

class VegaOffset(Filter):
    """
    VegaOffset: Class to compute AB-Vega offsets.

    USAGE = VEGA = VegaOffset([path=None],[filterName=Buser_V],[spectrumFile=A0V_Castelli.xml],[verbose=False])
    
          INPUT
                path -- Path to datasets repository. If None, will search for path in 
                        environment variables (stored as 'GALACTICUS_DATASETS'). 
                        [Default=None] 
          filterName -- Name of V-band filter. This name is used to construct the filter
                        file that will be loaded. [Default=Buser_V]
        spectrumFile -- Name of Vega spectrum file. [Default=A0V_Castelli.xml]
             verbose -- Print additional information. [Default=False] 

        Functions:
                 __call__(): Computes AB-Vega offset using computeOffset().
                 computeOffset(): Compute AB-Vega offset for specified filter 
                                  transmission curve.

    """    
    def __init__(self,path=None,filterName="Buser_V",spectrumFile="A0V_Castelli.xml",verbose=False):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(self.__class__,self).__init__()
        self.verbose = verbose
        DATA = GalacticusData(path=path,verbose=self.verbose)
        filterFile = DATA.search(filterName+".xml")
        self.VBAND = loadFilterFromFile(filterFile)
        self.fluxVega = None
        self.fluxAB = None
        self.VegaSpectrum = VegaSpectrum(path=path,fileName=spectrumFile,verbose=self.verbose)
        return

    def __call__(self,wavelength,transmission,kRomberg=8,**kwargs):
        return self.computeOffset(wavelength,transmission,kRomberg=kRomberg,**kwargs)
        
    def computeOffset(self,wavelength,transmission,kRomberg=8,**kwargs):
        """
        computeOffset(): Compute AB-Vega offset for specified filter transmission curve.

        USAGE: offset = VegaOffset().computeOffset(wavelength,transmission,[kRomberg=8],[**kwargs])
        
              INPUT
                 wavelength   -- Wavelengths for filter transmission curve.
                 transmission -- Transmission for filter transmission curve.
                 kRomberg     -- Number of k-nodes for Romberg integration. [Default=8]
                 **kwrgs      -- Keywords arguments to pass to scipy.interpolate.interp1d.
        
              OUTPUT
                 offset       -- AB-Vega offset for filter transmission curve.

        """
        # Compute fluxes for V-band if not already computed      
        if self.fluxAB is None or self.fluxVega is None:
            self.fluxAB,self.fluxVega = self.VegaSpectrum.computeFluxes(self.VBAND.transmission.wavelength,
                                                                        self.VBAND.transmission.transmission,\
                                                                            kRomberg=kRomberg,**kwargs)
        # Compute fluxes for specified filter
        fluxAB,fluxVega = self.VegaSpectrum.computeFluxes(wavelength,transmission,\
                                                              kRomberg=kRomberg,**kwargs)
        # Return Vega offset
        offset = 2.5*np.log10(fluxVega*self.fluxAB/self.fluxVega/fluxAB)
        return offset

