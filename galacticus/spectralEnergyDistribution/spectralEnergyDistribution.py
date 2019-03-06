#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import warnings
from .. import rcParams
from ..datasets import Dataset
from ..properties.manager import Property
from ..constants import megaParsec,centi,Pi,jansky,erg,luminosityAB,micro
from . import parseDatasetName,getSpectralEnergyDistributionWavelengths
from .continuum import sedContinuum
from .emissionLines import sedEmissionLines

@Property.register_subclass('spectralEneergyDistribution')
class SpectralEnergyDistribution(Property):

    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        self.Continuum = sedContinuum(self.galaxies)
        self.EmissionLines = sedEmissionLines(self.galaxies)
        return

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        SpectralEnergyDistribution.matches: Returns boolean to indicate whether this 
                                            class can process the specified property.
                                                                                                                                                                               
        USAGE:  match = SpectralEnergyDistribution.matches(propertyName,[redshift=None],[raiseError=False])

          INPUTS
               propertyName -- Name of property to process.
               redshift     -- Redshift value to query Galacticus HDF5 outputs. (Redundant in this case
                               but necessary for other properties.)
               raiseError   -- Raise error if property does not match.
                               (Default = False)

          OUTPUTS
                match       -- Boolean indicating whether this class can process 
                               this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid spectral energy distribution. "
            raise RuntimeError(msg)
        return False

    @classmethod
    def ergPerSecond(cls,sed):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        zeroCorrection = rcParams.getfloat("spectralEnergyDistribution",
                                           "zeroCorrection",
                                           fallback=1.0e-50)
        sed = np.log10(sed+zeroCorrection)
        sed += np.log10(luminosityAB)
        sed -= np.log10(erg)
        sed = 10.0**sed
        return sed

    def convertToMicroJanskies(self,redshift,sed):
        sed = self.ergPerSecond(sed)
        z = self.galaxies.get(redshift,properties=["redshift"])["redshift"].data
        comDistance = self.galaxies.GH5Obj.cosmology.comoving_distance(z)*megaParsec/centi
        comDistance = np.repeat(comDistance,sed.shape[1]).reshape(sed.shape)
        sed /= 4.0*Pi*comDistance**2
        sed /= jansky
        sed *= 1.0e6
        return sed

    def get(self,propertyName,redshift):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Extract information from property name
        MATCH = parseDatasetName(propertyName)
        # Get continuum 
        wavelengths,continuum = self.Continuum.get(propertyName,redshift)
        # Get emission lines
        if MATCH.group("noLines") is None:
            lines = self.EmissionLines.get(propertyName,redshift)
        else:
            lines = np.zeros_like(continuum)
        # Compute SED
        sed = self.convertToMicroJanskies(redshift,continuum+lines)
        DATA = Dataset(name=propertyName)
        attr = {"unitsInSI":jansky*micro*erg/centi**2}
        DATA.data = sed
        return DATA


