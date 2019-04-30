#! /usr/bin/env python
"""
galacticus.hydrogenGasDensity
=============================

Compute the hydrogen gas surface density for Galacticus galaxies.

"""

import sys,os,fnmatch,re
import numpy as np
import warnings
import unittest
from . import rcParams
from .datasets import Dataset
from .properties.manager import Property
from .constants import massSolar
from .constants import parsec,mega,centi,Pi
from .constants import massAtomic,atomicMassHydrogen,massFractionHydrogen

@Property.register_subclass('hydrogenGasDensity')
class HydrogenGasDensity(Property):
    """
    Compute the hydrogen gas density for a galaxy.
    
    Arguments:
        galaxies (object) : Instance of :class:`~galacticus.galaxies.Galaxies` class.
        verbose (bool,optional) : Print additional information

    Attributes:
        galaxies (object) : Instance of :class:`~galacticus.galaxies.Galaxies` class.
        verbose (bool) : Print additional information

    """
    
    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        return

    @classmethod
    def parseDatasetName(cls,datasetName):
        """
        Parse a hydrogen gas density dataset name.
        
        Arguments:
            datasetName (str) : Name of dataset to parse.

        Returns:
            match object : A `match <https://docs.python.org/3/library/re.html#re.Pattern.match>`_ instance.

        """
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Construct search string to pass to regex
        searchString = "^(?P<component>disk|spheroid)HydrogenGasDensity$"
        return re.search(searchString,datasetName)

    @classmethod
    def matches(cls,propertyName,redshift=None,raiseError=False):
        """
        Determines whether the specified property name matches the pattern requirements for a hydrogen gas density dataset name.

        Arguments:
            propertyName (str) : Dataset name to query.
            redshift (float,optional) : The redshift of the Galacticus snapshot output.
            raiseError (bool,optional) : Raise an error if dataset name does not match the pattern.

        Return:
            bool : Boolean indicating whether the dataset name matches the pattern.

        Raises:
            RuntimError : Raised if dataset name does not match pattern.

        """
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = cls.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid hydrogen gas density. "+\
                "Syntax is (disk|spheroid)HydrogenGasDensity."
            raise RuntimeError(msg)
        return False

    def getSurfaceDensityGas(self,component,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): requires either a 'disk' or 'spheroid' component.")
        # Extract gas mass and galaxy radius
        gas = component+"MassGas"
        radius = component+"Radius"
        GALS = self.galaxies.get(redshift,properties=[gas,radius])
        # Compute surface density in Mpc**2
        area = Pi*np.copy(GALS[radius].data)**2
        densitySurfaceGas = np.zeros_like(GALS[radius].data)
        mask = area>0.0
        densitySurfaceGas[mask] = np.copy(GALS[gas].data[mask]/area[mask])
        # Select method for computing density (central or mass-weighted)
        method = rcParams.get("hydrogenGasDensity","densityMethod",fallback="central")      
        if method.lower() == "central":
            densitySurfaceGas /= 2.0
        elif method.lower() == "massweighted":
            densitySurfaceGas /= 8.0
        else:
            msg = funcname+"(): in rcParams hydrogenGasDensty/densityMethod "+\
                "should be either 'central' of 'massWeighted'. Default=central."
            raise ValueError(msg)            
        return densitySurfaceGas

    @classmethod
    def getMassGiantMolecularClouds(cls):
        return rcParams.getfloat("hydrogenGasDensity","massGMC",fallback=3.7e+07)

    @classmethod
    def getCriticalSurfaceDensityClouds(cls):
        return rcParams.getfloat("hydrogenGasDensity","surfaceDensityCritical",fallback=8.5e13)
                
    def get(self,propertyName,redshift):
        """

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)
        component = MATCH.group('component')
        # Get gas surface density
        densitySurfaceGas = self.getSurfaceDensityGas(component,redshift)
        # Compute mass in GMCs
        massGMC = self.getMassGiantMolecularClouds()
        surfaceDensityCritical = self.getCriticalSurfaceDensityClouds()
        massClouds = np.zeros_like(densitySurfaceGas)
        mask = densitySurfaceGas > 0.0
        massClouds[mask] = massGMC/(densitySurfaceGas[mask]/surfaceDensityCritical)
        # Set surface density of clouds in Mpc**-2
        densitySurfaceClouds = np.maximum(densitySurfaceGas,surfaceDensityCritical)
        # Compute hydrogen density
        densityHydrogen = np.zeros_like(massClouds)
        mask = massClouds > 0.0
        densityHydrogen[mask] = (3.0/4.0)*np.sqrt(Pi)/np.sqrt(massClouds[mask])
        densityHydrogen *= densitySurfaceClouds**1.5
        densityHydrogen *= (centi/(mega*parsec))**3
        densityHydrogen *= massFractionHydrogen*massSolar
        densityHydrogen /= (massAtomic*atomicMassHydrogen)        
        # Create dataset
        DATA = Dataset(name=propertyName)
        attr = {"unitsInSI":centi**-3}
        attr["massGiantMolecularClouds"] = massGMC
        attr["criticalSurfaceDensityClouds"] = surfaceDensityCritical
        DATA.attr = attr
        DATA.data = np.copy(densityHydrogen)
        # Check if zero correction to be added
        zeroCorrection = rcParams.get("hydrogenGasDensity","zeroCorrection",fallback=None)
        if zeroCorrection is not None:
            DATA.data += float(zeroCorrection)
        del densityHydrogen,massClouds,densitySurfaceGas,densitySurfaceClouds
        return DATA


    
