#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import warnings
import unittest
from .. import rcParams
from ..datasets import Dataset
from ..properties.manager import Property
from ..Cloudy import CloudyTable
from ..filters.filters import GalacticusFilter
from ..constants import massSolar,luminositySolar,metallicitySolar
from ..constants import luminosityAB,erg
from ..constants import parsec,angstrom
from ..constants import mega,centi
from ..constants import Pi,speedOfLight
from ..constants import massAtomic,atomicMassHydrogen,massFractionHydrogen


def ergPerSecond(luminosity):
    luminosity *= luminositySolar/erg
    return luminosity


@Property.register_subclass('emissionLineLuminosity')
class EmissionLineLuminosity(Property):
    
    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        self.CLOUDY = CloudyTable()
        self.GALFIL = GalacticusFilter()
        return

    def lineInCloudyOutput(self,lineName):
        """
        EmissionLineLuminosity.lineInCloudyOutput: Returns boolean indicating whether specified
                                                   emission line can be found in CLOUDY output.

        USAGE:  result = EmissionLineLuminosity.lineInCloudyOutput(lineName)
        
          INPUTS
              lineName -- Name of emission line.

          OUTPUTS
              result   -- Boolean (T/F) indicating whether specified line is present.

        """
        return lineName in self.CLOUDY.listAvailableLines()

    def parseDatasetName(self,datasetName):
        """
        EmissionLineLuminosity.parseDatasetName: Parse an emission line luminosity 
                                                 dataset name.
        
        USAGE: SEARCH = EmissionLineLuminosity.parseDatasetName(propertyName)
        
             INPUTS
                propertyName -- Property name to parse.
                
             OUTPUTS
                SEARCH       -- Regex seearch (re.search) object or None if
                                propertyName cannot be parsed.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Construct search string to pass to regex
        searchString = "^(?P<component>disk|spheroid)LineLuminosity:"
        lines = "(?P<lineName>"+"|".join(self.CLOUDY.listAvailableLines())+")"
        searchString = searchString + lines + ":(?P<frame>rest|observed)"+\
            "(?P<filterName>:[^:]+)?"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            "(?P<recent>:recent)?$"
        return re.search(searchString,datasetName)
    
    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        EmissionLineLuminosity.matches: Returns boolean to indicate whether this 
                                        class can process the specified property.

        USAGE:  match = EmissionLineLuminosity.matches(propertyName,[redshift=None],
                                                       [raiseError=False])
                
          INPUTS 
               propertyName -- Name of property to process.
                   redshift -- Redshift value to query Galacticus HDF5 outputs. 
                               (Redundant in this particular case, but required 
                               for other properties.)
                raiseError  -- Raise error if property does not match. 
                               (Default = False)                

          OUTPUTS 
                match       -- Boolean indicating whether this class can process 
                               this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid emission line luminosity. "+\
                "Available emission lines: "+\
                ", ".join(self.CLOUDY.listAvailableLines())+"."
            raise RuntimeError(msg)
        return False

    def getContinuumLuminosityNames(self,propertyName):
        """
        EmissionLineLuminosity.getContinuumLuminosityNames: For specified emission line luminosity dataset,
                                                            return the name of the appopriate continuum
                                                            luminosity datasets.
        
        USAGE: LyName,HeName,OxName = EmissionLineLuminosity.getContinuumLuminosityNames(propertyName)
        
              INPUTS
                    propertyName -- Emission line dataset name
                    
              OUTPUTS
                    LyName       -- Lyman continuum luminosity dataset name
                    HeName       -- Helium continuum luminosity dataset name
                    OxName       -- Oxygen continuum luminosity dataset name
                    
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.matches(propertyName,raiseError=True)
        MATCH = self.parseDatasetName(propertyName)
        recent = ""
        if MATCH.group("recent"):
            recent = MATCH.group("recent")
        LymanName = MATCH.group("component")+"LymanContinuumLuminosity:"+MATCH.group("frame")+MATCH.group("redshiftString")+recent
        HeliumName = MATCH.group("component")+"HeliumContinuumLuminosity:"+MATCH.group("frame")+MATCH.group("redshiftString")+recent
        OxygenName = MATCH.group("component")+"OxygenContinuumLuminosity:"+MATCH.group("frame")+MATCH.group("redshiftString")+recent
        return LymanName,HeliumName,OxygenName

    def getContinuumLuminosities(self,propertyName,redshift):
        """
        EmissionLineLuminosity.getContinuumLuminosities: For specified emission line luminosity dataset,
                                                         return dictionary of appopriate continuum
                                                         luminosity datasets.
        
        USAGE: LUMINOSITIES = EmissionLineLuminosity.getContinuumLuminosities(propertyName,redshift)
        
              INPUTS
                    propertyName -- Emission line dataset name
                    redshift     -- Redshift to query Galacticus HDF5 outputs.
                    
              OUTPUTS
                    LUMINOSITIES -- Dictionary of Dataset() instances containing Lyman, Helium and 
                                    Oxygen continuuum luminosities. For any that are missing
                                    those dictionary entries will be set to None.
                    
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        LymanName,HeliumName,OxygenName = self.getContinuumLuminosityNames(propertyName)
        names = [LymanName,HeliumName,OxygenName]
        return self.galaxies.get(redshift,properties=names)

    @classmethod
    def getIonizingFluxHydrogen(cls,LyLuminosity):        
        """ionizingFluxHydrogen
        EmissionLineLuminosity.getIonizingFluxHydrogen(): Compute the Lyman ionizing flux from the supplied
                                                          Lyman continuuum luminosity.

        USAGE: flux = EmissionLineLuminosity.getIonizingFluxHydrogen(luminosity)

           INPUTS
              luminosity -- Lyman continuum luminosity
              
           OUTPUTS
              flux       -- log10 of Lyman ionizing flux.

        """
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name        
        ionizingFluxHydrogen = np.copy(LyLuminosity)
        mask = ionizingFluxHydrogen==0.0
        if any(mask):
            ionizingFluxHydrogen[mask] = 1.0e-50
        del mask
        return np.log10(ionizingFluxHydrogen)+50.0
    
    @classmethod
    def getIonizingFluxRatio(cls,LyLuminosity,XLuminosity):
        """
        EmissionLineLuminosity.getIonizingFluxRatio(): Compute the ratio of Helium or Oxygen ionizing flux 
                                                       to Lyman ionizing flux.

        USAGE: ratio = EmissionLineLuminosity.getIonizingFluxRatio(LymanLuminosity,XLuminosity)

           INPUTS
              LymanLuminosity -- Lyman continuum luminosity.
              XLuminosity     -- Helium or Oxygen luminosity.
              
           OUTPUTS
              ratio           -- log10 of ionizing flux ratio..

        """
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        LymanFlux = np.copy(LyLuminosity)        
        np.place(LymanFlux,LymanFlux==0.0,np.nan)
        XFlux = np.copy(XLuminosity)
        np.place(XFlux,XFlux==0.0,np.nan)
        return np.log10(XFlux/LymanFlux)

    @classmethod
    def getMassHIIRegions(cls):
        """
        EmissionLineLuminosity.getMassHIIRegion(): Return the mass of HII regions, as stored in 
                                                   the configuration parameters. Default value 
                                                   is 7.5e3 Msol.
                                                   
        USAGE: mass = EmissionLineLuminosity.getMassHIIRegion()

           OUTPUTS
               mass -- Mass of HII regions in Solar masses.

        """
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        return rcParams.getfloat("emissionLine","massHIIRegion",fallback=7.5e3)
    
    @classmethod
    def getLifetimeHIIRegions(cls):
        """
        EmissionLineLuminosity.getLifetimeHIIRegion(): Return the lifetime of HII regions, as stored in 
                                                       the configuration parameters. Default value is 
                                                       1.0e-3 Gyr.
                                                   
        USAGE: mass = EmissionLineLuminosity.getLifetimeHIIRegion()

           OUTPUTS
               mass -- Lifetime of HII regions in Gyrs.

        """
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        return rcParams.getfloat("emissionLine","lifetimeHIIRegion",fallback=1.0e-3)

    def getNumberHIIRegions(self,redshift,component):
        """
        EmissionLineLuminosity.getNumberHIIRegions(): Return number of HII regions in for specified
                                                      galaxy component at specified redshift.

        USAGE: number = EmissionLineLuminosity.getNumberHIIRegions(redshift,component)
        
           INPUTS
               redshift  -- Redshift value to query Galacticus HDF5 outputs.
               component -- String indicating component to compute number for. String
                            be either 'disk' or 'spheroid'.

           OUTPUT
               number   -- Number of HII regions.      
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): Component '"+component+"' not recognized. "+\
                                 "Should be either 'disk' or 'spheroid'.")
        sfrName = component+"StarFormationRate"
        GALS = self.galaxies.get(redshift,properties=[sfrName])
        massHIIRegion = self.getMassHIIRegions()
        lifetimeHIIRegion = self.getLifetimeHIIRegions()
        return GALS[sfrName].data*lifetimeHIIRegion/massHIIRegion
        
    def getLuminosityMultiplier(self,propertyName,redshift):
        """
        EmissionLineLuminosity.getLuminosityMultiplier(): If emission line is under a broadband filter,
                                                          compute the multiplicative factor to convert 
                                                          line luminosity to broadband luminosity in 
                                                          AB units. If line is not under a filter, i.e.
                                                          no filter is specified in the properrty name,
                                                          then the multiplication factor is unity.

        USAGE: multiplier = EmissionLineLuminosity.getLuminosityMultiplier(propertyName,redshift)

          INPUTS
               propertyName -- Property name to compute multiplier for. Should be a
                               valid emission line luminosity dataset name.
               redshift     -- Redshift value to query Galacticus HDF5 outputs.
         

          OUTPUTS
               multiplier   -- Multiplication factor. Equal to unity if no filter
                               specified.


        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)
        # Check if filter name provided
        filterName = MATCH.group('filterName')
        # Exit if no filter provided
        if filterName is None:
            return 1.0
        # Compute multiplier for filter
        # Load Filter instance
        FILTER = self.GALFIL.load(filterName.replace(":",""))
        # Get the redshift of the galaxies
        GALS = self.galaxies.get(redshift,properties=["redshift"])
        # Extract frame (rest or observed)
        frame = MATCH.group('frame').replace(":","")
        # Get line wavelength
        lineWavelength = self.CLOUDY.getWavelength(MATCH.group('lineName'))
        if frame == "observed":            
            lineWavelength *= (1.0+GALS['redshift'].data)
        else:
            lineWavelength *= np.ones_like(GALS['redshift'].data)
        # Interpolate the transmission to the line wavelength
        multiplier = FILTER.interpolate(lineWavelength)
        # Compute the multiplicative factor to convert line
        # luminosity to luminosity in AB units in the filter
        multiplier /= FILTER.integrate()
        # Galacticus defines observed-frame luminosities
        # by simply redshifting the galaxy spectrum without
        # changing the amplitude of F_nu (i.e. the compression of
        # the spectrum into a smaller range of frequencies is not
        # accounted for). For a line, we can understand how this
        # should affect the luminosity by considering the line as
        # a Gaussian with very narrow width (such that the full
        # extent of the line always lies in the filter). In this
        # case, when the line is redshifted the width of the
        # Gaussian (in frequency space) is reduced, while the
        # amplitude is unchanged (as, once again, we are not
        # taking into account the compression of the spectrum into
        # the smaller range of frequencies). The integral over the
        # line will therefore be reduced by a factor of (1+z) -
        # this factor is included in the following line. Note
        # that, when converting this observed luminosity into an
        # observed flux a factor of (1+z) must be included to
        # account for compression of photon frequencies (just as
        # with continuum luminosities in Galacticus) which will
        # counteract the effects of the 1/(1+z) included below.
        if frame == "observed":
            multiplier /= (1.0+GALS["redshift"].data)
        else:
            lineWavelength *= np.ones_like(GALS['redshift'].data)
        return multiplier

    def get(self,propertyName,redshift):
        """
        EmissionLineLuminosity.get(): Compute specified emission line luminosity at specified
                                      redshift. 

        USAGE: DATA = EmissionLineLuminosity.get(propertyName,redshift)

           INPUTS
               propertyName -- Property name to compute luminosity for. Should be a
                               valid emission line luminosity dataset name.
               redshift     -- Redshift value to query Galacticus HDF5 outputs.
        
           OUTPUTS
               DATA         -- Dataset() class instance containing luminosity information, or
                               None if line luminosity cannot be computed.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)        
        # Get continuum luminosity names
        LymanName,HeliumName,OxygenName = self.getContinuumLuminosityNames(propertyName)
        # Extract continuum luminosities
        FLUXES = self.getContinuumLuminosities(propertyName,redshift)
        # If any luminosities are missing, unable to compute emission lines so return 'None' instance.
        if any([FLUXES[name] is None for name in FLUXES.keys()]):
            warnings.warn(funcname+"(): Unable to compute emission line luminosity as one of the "+\
                              "continuum luminosities is missing. Returning None instance.")
            return None
        # Get galaxy properties for calculation
        metals = MATCH.group('component')+"GasMetallicity"
        hydrogen = MATCH.group('component')+"HydrogenGasDensity"
        GALS = self.galaxies.get(redshift,properties=[metals,hydrogen])
        # i) Hydrogen gas density
        hydrogenGasDensity = np.log10(np.copy(GALS[hydrogen].data))
        # ii) Metallicity
        metallicity = np.log10(np.copy(GALS[metals].data))
        del GALS
        # iii) Ionizing Hydrogen flux
        ionizingFluxHydrogen = self.getIonizingFluxHydrogen(FLUXES[LymanName].data)
        # Convert the hydrogen ionizing luminosity to be per HII region
        numberHIIRegion = self.getNumberHIIRegions(redshift,MATCH.group('component'))
        mask = numberHIIRegion == 0.0
        numberHIIRegion[mask] = 1.0
        ionizingFluxHydrogen -= np.copy(np.log10(numberHIIRegion))
        numberHIIRegion[mask] = 0.0
        # iv) Luminosity ratios He/H and Ox/H 
        ionizingFluxHeliumToHydrogen = self.getIonizingFluxRatio(FLUXES[LymanName].data,FLUXES[HeliumName].data)
        ionizingFluxOxygenToHydrogen = self.getIonizingFluxRatio(FLUXES[LymanName].data,FLUXES[OxygenName].data)
        # Truncate properties to table bounds where necessary to avoid unphysical extrapolations.
        #
        ## Hydrogen density and H-ionizing flux are truncated at both the lower and upper extent
        ## of the tabulated range. The assumption is that the table covers the plausible
        ## physical range for these values. In the case of H-ionizing flux, if we truncate the
        ## value we must then apply a multiplicative correction to the line luminosity to ensure
        ## that we correctly account for all ionizing photons produced.
        ##
        ## For metallicity and the He/H and O/H ionizing flux ratios we truncate only at the
        ## upper extent of the tabulated range. The table is assumed to be tabulated up to the
        ## maximum physically plausible extent for these quantities. Extrapolation to lower
        ## values should be reasonably robust (and the table is assumed to extend to
        ## sufficiently low values that the consequences of extrapolation are unlikely to be
        ## observationally relevant anyway).
        hydrogenGasDensityTable           = self.CLOUDY.getInterpolant('densityHydrogen'             )
        metallicityTable                  = self.CLOUDY.getInterpolant('metallicity'                 )
        ionizingFluxHydrogenTable         = self.CLOUDY.getInterpolant('ionizingFluxHydrogen'        )
        ionizingFluxHeliumToHydrogenTable = self.CLOUDY.getInterpolant('ionizingFluxHeliumToHydrogen')
        ionizingFluxOxygenToHydrogenTable = self.CLOUDY.getInterpolant('ionizingFluxOxygenToHydrogen')
        boundLowHydrogenGasDensity            = hydrogenGasDensity           < hydrogenGasDensityTable          [ 0]
        boundHighHydrogenGasDensity           = hydrogenGasDensity           > hydrogenGasDensityTable          [-1]
        boundHighMetallicity                  = metallicity                  > metallicityTable                 [-1]
        boundLowIonizingFluxHydrogen          = ionizingFluxHydrogen         < ionizingFluxHydrogenTable        [ 0]
        boundHighIonizingFluxHydrogen         = ionizingFluxHydrogen         > ionizingFluxHydrogenTable        [-1]
        boundHighIonizingFluxHeliumToHydrogen = ionizingFluxHeliumToHydrogen > ionizingFluxHeliumToHydrogenTable[-1]
        boundHighIonizingFluxOxygenToHydrogen = ionizingFluxOxygenToHydrogen > ionizingFluxOxygenToHydrogenTable[-1]
        ionizingFluxMultiplier                = np.zeros(ionizingFluxHydrogen.shape)
        hydrogenGasDensity          [boundLowHydrogenGasDensity           ] =  hydrogenGasDensityTable          [ 0]
        hydrogenGasDensity          [boundHighHydrogenGasDensity          ] =  hydrogenGasDensityTable          [-1]
        metallicity                 [boundHighMetallicity                 ] =  metallicityTable                 [-1]
        ionizingFluxMultiplier      [boundLowIonizingFluxHydrogen         ] = -ionizingFluxHydrogenTable        [ 0]+ionizingFluxHydrogen[boundLowIonizingFluxHydrogen ]
        ionizingFluxMultiplier      [boundHighIonizingFluxHydrogen        ] = -ionizingFluxHydrogenTable        [-1]+ionizingFluxHydrogen[boundHighIonizingFluxHydrogen]
        ionizingFluxHydrogen        [boundLowIonizingFluxHydrogen         ] =  ionizingFluxHydrogenTable        [ 0]
        ionizingFluxHydrogen        [boundHighIonizingFluxHydrogen        ] =  ionizingFluxHydrogenTable        [-1]
        ionizingFluxHeliumToHydrogen[boundHighIonizingFluxHeliumToHydrogen] =  ionizingFluxHeliumToHydrogenTable[-1]
        ionizingFluxOxygenToHydrogen[boundHighIonizingFluxOxygenToHydrogen] =  ionizingFluxOxygenToHydrogenTable[-1]
        # Create Dataset() instance        
        DATA = Dataset(name=propertyName)
        attr = {"unitsInSI":luminositySolar}
        attr["massHIIRegion"] = self.getMassHIIRegions()
        attr["lifetimeHIIRegion"] = self.getLifetimeHIIRegions()
        DATA.attr = attr        
        # Pass properties to CloudyTable() class for interpolation        
        DATA.data = np.copy(self.CLOUDY.interpolate(MATCH.group("lineName"),metallicity,hydrogenGasDensity,\
                                                        ionizingFluxHydrogen,ionizingFluxHeliumToHydrogen,\
                                                        ionizingFluxOxygenToHydrogen))        
        # Clear memory
        del metallicity,hydrogenGasDensity,ionizingFluxHydrogen
        del ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen
        # Get luminosity multiplier
        luminosityMultiplier = self.getLuminosityMultiplier(propertyName,redshift)
        # Convert units of luminosity 
        DATA.data *= (10.0**ionizingFluxMultiplier*luminosityMultiplier*numberHIIRegion*erg/luminositySolar)
        # Offset zero luminosities
        zeroCorrection = rcParams.getfloat("emissionLine","zeroCorrection",fallback=1.0e-50)
        DATA.data += zeroCorrection
        return DATA


