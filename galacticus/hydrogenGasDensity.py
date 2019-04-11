#! /usr/bin/env python

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
    
    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        return

    @classmethod
    def parseDatasetName(cls,datasetName):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Construct search string to pass to regex
        searchString = "^(?P<component>disk|spheroid)HydrogenGasDensity$"
        return re.search(searchString,datasetName)

    @classmethod
    def matches(cls,propertyName,redshift=None,raiseError=False):
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
        massClouds = massGMC/(densitySurfaceGas/surfaceDensityCritical)
        # Set surface density of clouds in Mpc**-2
        densitySurfaceClouds = np.maximum(densitySurfaceGas,surfaceDensityCritical)
        # Compute hydrogen density
        densityHydrogen = (3.0/4.0)*np.sqrt(Pi)/np.sqrt(massClouds)
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
        del densityHydrogen,massClouds,densitySurfaceGas,densitySurfaceClouds
        return DATA


    
