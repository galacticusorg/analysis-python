#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import unittest
from ..datasets import Dataset
from ..properties.manager import Property
from ..Cloudy import CloudyTable
from ..filters import filterLuminosityAB
from ..filters.filters import GalacticusFilters
from ..constants import massSolar,luminositySolar,metallicitySolar
from ..constants import luminosityAB,erg
from ..constants import arsec,angstrom
from ..constants import mega,centi
from ..constants import Pi,speedOfLight
from ..constants import massAtomic,atomicMassHydrogen,massFractionHydrogen

@Property.register_subclass('emissionLineLuminosity')
class EmissionLineLuminosity(Property):
    
    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        self.CLOUDY = CloudyTable()
        self.GALFIL = GalacticusFilters()
        return
    
    def parseDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract dataset name information
        searchString = "^(?P<component>disk|spheroid)LineLuminosity:(?P<lineName>[^:]+)(?P<frame>:[^:]+)"+\
            "(?P<filterName>:[^:]+)?(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<recent>:recent)?$"
        return re.search(searchString,datasetName)
    
    def matches(self,propertyName,redshift=None):
        if self.parseDatasetName(propertyName):
            return True
        return False
        
    def getHydrogenGasDensity(self,redshift,component):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract gas mass and galaxy radius
        gas = component+"MassGas"
        radius = component+"Radius"
        GALS = self.galaxies.get(redshift,properties=[gas,radius])       
        # Compute volume in cm^3
        volume = np.copy((GALS[radius].data*mega*parsec/centi)**3)
        np.place(volume,volume==0.0,np.nan)
        # Compute gas mass in kg
        mass = np.copy(GALS[gas])*massSolar
        np.place(mass,mass==0.0,np.nan)
        # Clear GALS from memory
        del GALS
        # Compute density in kg/cm^3        
        density = np.copy(mass/volume)
        density *= massFractionHydrogen
        density /= (4.0*Pi*massAtomic*atomicMassHydrogen)
        density = np.log10(density)
        # Clear memory from unnecessary variables
        del mass,volume
        return density

    def getIonizingFluxHydrogen(self,LyContinuumName,redshift):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        GALS = self.galaxies.get(redshift,properties=[LyContinuumName])
        if GALS[LyContinuumName] is None:
            raise ValueError(funcname+"(): Lyman Continuum luminosity returned 'None' instance.")
        ionizingFluxHydrogen = np.copy(GALS[LyContinuumName].data)
        del GALS
        np.place(ionizingFluxHydrogen,ionizingFluxHydrogen==0.0,np.nan)
        return np.log10(ionizingFluxHydrogen)+50.0
    
    def getIonizingFluxRatio(self,LyContinuumName,XContinuumName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        GALS = self.galaxies.get(redshift,properties=[LyContinuumName])
        if GALS[LyContinuumName] is None:
            raise ValueError(funcname+"(): Lyman Continuum luminosity returned 'None' instance.")
        if GALS[XContinuumName] is None:
            if "Helium" in XContinuumName:
                name = "Helium"
            else:
                name = "Oxygen"            
            raise ValueError(funcname+"(): "+name+" Continuum luminosity returned 'None' instance.")
        LymanFlux = np.copy(GALS[LyContinuumName])
        np.place(LymanFlux,LymanFLux==0.0,np.nan)
        XFlux = np.copy(GALS[XContinuumName])
        np.place(XFlux,XFLux==0.0,np.nan)
        del GALS
        return np.log10(XFlux/LyFLux)

    def getMassHIIRegions(self):
        return rcParams.getFloat("emissionLine","massHIIRegion",fallback=7.5e3)
    
    def getLifetimeHIIRegions(self):
        return rcParams.getFloat("emissionLine","lifetimeHIIRegion",fallback=1.0e-3)

    def getNumberHIIRegions(self,redshift,component):
        sfrName = component+"StarFormationRate"
        GALS = self.galaxies.get(redshift,properties=[sfrName])
        massHIIRegion = self.getMassHIIRegions()
        lifetimeHIIRegion = self.getLifetimeHIIRegions()
        return GALS[sfrName].data*lifetimeHIIRegion/massHIIRegion

    def getContinuumLuminosityNames(self,propertyName):
        MATCH = self.parseDatasetName(propertyName)
        recent = ""
        if MATCH.group("recent"):
            recent = MATCH.group("recent")
        LymanContinuuum = MATCH.group("component")+"LymanContinuumLuminosity"+MATCH.group("redshiftString")+recent
        HeliumContinuuum = MATCH.group("component")+"HeliumContinuumLuminosity"+MATCH.group("redshiftString")+recent
        OxygenContinuuum = MATCH.group("component")+"OxygenContinuumLuminosity"+MATCH.group("redshiftString")+recent
        return LymanContinuuum,HeliumContinuuum,OxygenContinuuum
            
    def getLuminosityMultiplier(self,porpertyName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not an emission line luminosity."
            raise RuntimeError(msg)
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)
        # Check if filter name provided
        filterName = MATCH.group('filterName')
        # Exit if no filter provided
        if filterName is None:
            return 1.0
        # Compute multiplier for filter
        # Load Filter instance
        FILTER = self.GALFIL.load(filterName)
        # Get the redshift of the galaxies
        GALS = self.galaxies.get(redshift,properties=["redshift"])
        # Extract frame (rest or observed)
        frame = MATCH.group('frame').replace(":","")
        # Get line wavelength
        lineWavelength = self.CLOUDY.getWavelength(MATCH.group('lineName'))
        if frame == "observed":            
            lineWavelength *= (1.0+GALS['redshift'].data)
        # Interpolate the transmission to the line wavelength
        multiplier = FILTER.getTransmissionAtWavelength(lineWavelength)
        # Compute the multiplicative factor to convert line
        # luminosity to luminosity in AB units in the filter
        multiplier /= filterLuminosityAB(FILTER,k=10)
        # Correct multiplier for redshift
        if frame == "observed":
            multiplier /= (1.0+GALS["redshift"].data)
        return multiplier

    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not an emission line luminosity."
            raise RuntimeError(msg)
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)
        # Get galaxy properties for calculation
        # i) Hydrogen gas density
        hydrogenGasDensity = self.getHydrogenGasDensity(redshift,MATCH.grooup('component'))
        # ii) Metallicity
        metals = MATCH.group('component')+"Metallicity"
        GALS = self.galaxies.get(redshift,properties=[metals])
        metallicity = np.copy(GALS[metals].data)
        del GALS
        # Get continuum luminosity names
        LymanContinuuum,HeliumContinuuum,OxygenContinuuum = self.getContinuumLuminosityNames(propertName)
        # iii) Ionizing Hydrogen flux
        ionizingFluxHydrogen = self.getIonizingFluxHydrogen(LymanContinuum,redshift)
        # Convert the hydrogen ionizing luminosity to be per HII region
        numberHIIRegion = self.getNumberHIIRegions(redshift,MATCH.group('component'))
        np.place(numberHIIRegion,numberHIIRegion==0.0,np.nan)
        ionizingFluxHydrogen -= np.log10(numberHIIRegion)
        # iv) Luminosity ratios He/H and Ox/H 
        ionizingFluxHeliumToHydrogen = self.getIonizingFluxRatio(LymanContinuum,HeliumContinuum,redshift)
        ionizingFluxOxygenToHydrogen = self.getIonizingFluxRatio(LymanContinuum,OxygenContinuum,redshift)
        # Create Dataset() instance        
        DATA = Dataset(name=propertyName)
        attr = {"unitsInSI":luminositySolar}
        attr["massHIIRegion"] = self.getMassHIIRegions()
        attr["lifetimeHIIRegion"] = self.getLifetimeHIIRegions()
        DATA.attr = attr
        # Pass properties to CloudyTable() class for interpolation        
        DATA.data = np.copy(self.CLOUDY.interpolate(MATCH.group("lineName"),metallicity,densityHydrogen,\
                                                        ionizingFluxHydrogen,ionizingFluxHeliumToHydrogen,\
                                                        ionizingFluxOxygenToHydrogen))
        # Clear memory
        del metallicity,densityHydrogen,ionizingFluxHydrogen
        del ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen
        # Get luminosity multiplier
        luminosityMultiplier = self.getLuminosityMultiplier()
        # Convert units of luminosity 
        DATA.data *= (luminosityMultiplier*numberHIIRegion*erg/luminositySolar)
        return DATA

