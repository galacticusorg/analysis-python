#! /usr/bin/env python

import sys,os,re
import numpy as np
import unittest
import copy
from .. import rcParams
from ..datasets import Dataset
from ..properties.manager import Property
from ..constants import Pi,megaParsec,milli,centi,kilo
from ..constants import massAtomic,massSolar,massFractionHydrogen
from .CompendiumTable import CompendiumTable

@Property.register_subclass('dustOpticalDepthCentral')
class DustOpticalDepthCentral(Property):
    """
    DustOpticalDepthCentral(): Compute dust optical depths through the centers of galaxy disks.

    Methods:
           parseDatasetName(): Parse optical depth dataset names
           matches(): Indicates whether specified dataset can be processed by this class.
           computeColumnDensityMetals(): Compute column density of metals in galaxy disks.
           getOpacity(): Return the opacity either from compendium file or by approximation.
           get(): Compute dust optical depths through centers of galaxy disks at specified reshift.

    """
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def parseDatasetName(self,propertyName):
        """
        DustOpticalDepthCentral.parseDatasetName(): Parse a dust optical depth dataset.

        USAGE:  SEARCH = DustOpticalDepthCentral.parseDatasetName(propertyName)

           INPUTS
                propertyName -- Property name to parse.

           OUTPUTS
                SEARCH       -- Regex seearch (re.search) object or None if
                                propertyName cannot be parsed.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        searchString = "^(?P<component>disk|spheroid)DustOpticalDepthCentral:dust(?P<dust>Atlas|Compendium)$"
        MATCH = re.search(searchString,propertyName)
        if MATCH is not None:
            return MATCH
        return None

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        DustOpticalDepthCentral.matches(): Returns boolean to indicate whether this
                                        class can process the specified property.
        
        USAGE: matches = DustOpticalDepthCentral.matches(propertyName,[redshift=None],
                                                         [raiseError=False])

           INPUTS 
               propertyName -- Name of property to process.
                   redshift -- Redshift value to query Galacticus HDF5 outputs.  
                               (Redundant in this particular case, but required 
                               for other properties.)  
                 raiseError -- Raise error if property does not match. (Default = False)

          OUTPUTS
                match       -- Boolean indicating whether this class can process
                               this property.   

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+"' is not valid."+\
                " Use 'diskDustOpticalDepthCentral:dust(Atlas|Compendium)'."
            raise RuntimeError(msg)
        return False
    
    def computeColumnDensityMetals(self,redshift,component):
        """
        DustOpticalDepthCentral.computeColumnDensityMetals(): Compute the column density of metals in the
                                                              disk of the galaxy in Msol/Mpc^2.

        USAGE: density = DustOpticalDepthCentral.computeColumnDensityMetals(redshift)

            INPUT
               redshift  -- Redshift value to query Galacticus HDF5 outputs.  
               component -- Galaxy component ('disk' or 'spheroid').
            OUTPUT
               density   -- Numpy array of column densities.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): Component must be 'disk' or 'spheroid'!")
        metalsName = component+"AbundancesGasMetals"
        radiusName = component+"Radius"
        PROPS = self.galaxies.get(redshift,properties=[metalsName,radiusName])
        columnDensityMetals = np.zeros_like(PROPS[metalsName].data)
        mask = np.logical_and(PROPS[radiusName].data>0.0,PROPS[metalsName].data>=0.0)
        columnDensityMetals[mask] = np.copy(PROPS[metalsName].data[mask])
        columnDensityMetals[mask] /= (2.0*Pi*np.copy(PROPS[radiusName].data[mask])**2)
        return columnDensityMetals

    def getOpacity(self,dustLabel):        
        """
        DustOpticalDepthCentral.getOpacity(): Return the opacity of through the center of the galaxy.

        USAGE:  opacity = DustOpticalDepthCentral.getOpacity(dustLabel)

            INPUT  
                dustLabel -- String to indicate dust method to use ('Atlas' or 'Compendium')

            OUTPUTS
                opacity   -- Numpy array storing opacities of galaxies.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if dustLabel == "Compendium":
            # Get opacity in cm^2/g
            COMPENDIUM = CompendiumTable()
            COMPENDIUM.loadOpacity()
            opacity = copy.copy(COMPENDIUM.opacity)
            del COMPENDIUM
            # Get the dust-to-metals ratio. If not provided we use a
            # default of 0.44 which is approximately correct for the
            # Milky Way (e.g. Popping et al.; 2017;
            # http://adsabs.harvard.edu/abs/2017MNRAS.471.3152P).
            dustToMetalsRatio = rcParams.getfloat("dustOpticalDepth","dustToMetalsRatio",fallback=0.44)
            opacity *= dustToMetalsRatio
        elif dustLabel == "Atlas":
            # Approximate opacity
            # i) specify necessary parameters
            localISMMetallicity = rcParams.getfloat("dustOpticalDepth","localISMMetallicity",fallback=0.02)
            AV_to_EBV = 3.10            # ... (A_V/E(B-V); Savage & Mathis 1979)
            NH_to_EBV = 5.8e21          # ... (N_H/E(B-V); atoms/cm^2/mag; Savage & Mathis 1979)                                                                               
            opticalDepthToMagnitudes = 2.5*np.log10(np.exp(1.0))
            # ii) compute opacity in cm^2/g
            opacity = (AV_to_EBV/opticalDepthToMagnitudes)/NH_to_EBV
            opacity *= (massFractionHydrogen/(massAtomic*kilo))
            #     Rescale by local ISM metallicity
            opacity /= localISMMetallicity
        else:
            raise ValueError(funcname+"(): Dust label '"+dustLabel+"' not recognized. "+\
                                 "Should be 'Atlas' or 'Compendium'.")
        return opacity

    def get(self,propertyName,redshift):
        """
        DustOpticalDepthCentral.get(): Return the dust optical depth through the center of the
                                       galactic disk.
        
        USAGE:  DATA = DustOpticalDepthCentral.get(propertyName,redshift)

            INPUTS
               propertyName -- Name of property to compute. This should be set to 
                               '(disk|spheroid)DustOpticalDepthCentral:dust(Atlas|Compendium)'.
               redshift     -- Redshift value to query Galacticus HDF5 outputs.                                                                                                
        
            OUTPUT
               DATA         -- Instance of galacticus.datasets.Dataset() class                                                                                                 
                               containing computed galaxy information.    
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        # Get column density for metals
        columnDensityMetals = self.computeColumnDensityMetals(redshift,MATCH.group('component'))               
        # Correct column density units to g/cm^2
        columnDensityMetals *= (massSolar*kilo)*(centi/megaParsec)**2
        # Get opacity in cm^2/g
        opacity = self.getOpacity(MATCH.group("dust")) 
        # Compute optical depth
        DATA = Dataset(name=propertyName)
        DATA.data = np.copy(columnDensityMetals*opacity)
        return DATA

