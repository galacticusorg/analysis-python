#! /usr/bin/env python

import sys
import numpy as np
import fnmatch
from .. import rcParams
from ..io import GalacticusHDF5
from .lineProfiles import LineProfiles
from ..Cloudy import CloudyTable
from ..constants import speedOfLight,angstrom
from . import getSpectralEnergyDistributionWavelengths,parseDatasetName

class EmissionLines(object):

    def __init__(self,galaxies):
        self.galaxies = galaxies
        self.CLOUDY = CloudyTable()
        self.CLOUDY.loadEmissionLines()
        self.profile = rcParams.get("emissionLine","profileShape",fallback='gaussian')
        return

    def getLineLuminosity(self,MATCH,redshift,lineName):
        datasetName = MATCH.group('component')+"LineLuminosity:"+lineName+\
                      ":"+MATCH.group("frame")+MATCH.group("redshiftString")
        if MATCH.group("recent") is not None:
            datasetName = datasetName + MATCH.group("recent")
        if MATCH.group("dust") is not None:
            datasetName = datasetName + MATCH.group("dust")
        DATA = self.galaxies.get(redshift,properties=[datasetName])
        return DATA[datasetName].data
        
    def addLineProfile(self,LINE,MATCH,redshift,wavelengths,luminosities):
        # Get line luminosity
        lineLuminosity = self.getLineLuminosity(MATCH,redshift,LINE.name)        
        # Get line wavelength
        lineWavelength = np.ones_like(lineLuminosity)*LINE.wavelength
        if fnmatch.fnmatch(MATCH.group("frame"),"observed"):
            z = self.galaxies.get(redshift,properties=["redshift"])["redshift"].data
            lineWavelength *= (1.0+z)
        # Get FWHM
        if MATCH.group("lineWidth") is None:
            width = "dispersionWidth"
        else:
            width = MATCH.group("lineWidth").replace(":","")
        datasetName = "fullWidthHalfMaximum:"+LINE.name+":"+width+\
                      MATCH.group("redshiftString")
        if MATCH.group("recent") is not None:
            datasetName = datasetName + MATCH.group("recent")
        FWHM = self.galaxies.get(redshift,properties=[datasetName])[datasetName].data
        if fnmatch.fnmatch(MATCH.group("frame"),"observed"):
            z = self.galaxies.get(redshift,properties=["redshift"])["redshift"].data
            FWHM *= (1.0+z)
       # Apply line profile
        if fnmatch.fnmatch(self.profile.lower(),"gaussian"):
            luminosities += LineProfiles.gaussian(wavelengths,lineWavelength,lineLuminosity,FWHM)
        return

    def sumLineProfiles(self,propertyName,redshift):
        MATCH = parseDatasetName(propertyName)
        # Extract wavelengths for SED
        wavelengths = getSpectralEnergyDistributionWavelengths(propertyName)
        # Create empty array to store line luminosities
        z = self.galaxies.get(redshift,properties=["redshift"])["redshift"].data
        luminosities = np.zeros((len(z),len(wavelengths)),dtype=float)
        # Add in the luminosities for the individual lines
        [self.addLineProfile(LINE,MATCH,redshift,wavelengths,luminosities) 
         for LINE in self.CLOUDY.lines.values()]
        # Convert units
        frequency = speedOfLight/np.stack([wavelengths]*luminosities.shape[0])*angstrom
        luminosities /= frequency
        return luminosities
        
    def get(self,propertyName,redshift):
        luminosities = self.sumLineProfiles(propertyName,redshift)
        return luminosities
