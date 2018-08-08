#! /usr/bin/env python

import sys
import re
import numpy as np
import scipy.interpolate
import h5py
from . import rcParams
from .datasets import Dataset
from .properties.manager import Property
from .constants import megaParsec, massSolar, centi, milli
from .filters import Filter
from .filters.filters import GalacticusFilter
from .data import GalacticusData

@Property.register_subclass('dustCompendium')
class DustCompendium(Property):
    """
    DustCompendium: Compute dust-extinguished luminosities using the dust compendium tabulations.

    Functions:
            matches(): Indicates whether specified dataset can be processed by this class.
            get(): Computes dust-extinguished luminosities at specified redshift.

    """    
    dustCompendiumRegEx = "^(?P<component>disk|spheroid)LuminositiesStellar:(?P<filter>[^:]+):(?P<frame>[^:]+)"+\
                          "(?P<redshiftString>:z(?P<redshift>[\d\.]+)):dustCompendium"

    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies     = galaxies
        self.data         = GalacticusData(verbose=False)
        self.tablesLoaded = False
        return

    def matches(self,propertyName,redshift=None):
        """
        DustCompendium.matches(): Returns boolean to indicate whether this class can process
                                 the specified property.

        USAGE: match =  DustCompendium.matches(propertyName,[redshift=None])

          INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.                           

          OUTPUTS
              match        -- Boolean indicating whether this class can process 
                              this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        isMatched = re.search(self.dustCompendiumRegEx,propertyName) or propertyName == "tauV0:dustCompendium"
        return isMatched        

    def get(self,propertyName,redshift):        
        """
        DustCompendium.get(): Compute dust-extinguished luminosities for specified redshift.
        
        USAGE:  DATA = DustCompendium.get(propertyName,redshift)
                
           INPUTS
           
                propertyName -- Name of property to compute.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.
           
           OUTPUT
                DATA         -- Instance of galacticus.datasets.Dataset() class containing 
                                computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if propertyName == "tauV0:dustCompendium":
            # If the central optical depth is requested we only need intrinsic disk properties, and force the component to "disk"
            # to avoid considering spheroid properties later.
            properties = self.galaxies.get(redshift,properties=[ "diskAbundancesGasMetals", "diskRadius" ])
            component = "disk"
        else:
            # Any other property we receive should be a dust-attenuated luminosity.
            propertyMatch = re.search(self.dustCompendiumRegEx,propertyName)
            if not propertyMatch:
                raise RuntimeError(funcname+"(): Cannot process property '"+propertyName+"'.")
            redshift       = float(propertyMatch.group("redshift" ))
            filterName     =       propertyMatch.group("filter"   )
            frame          =       propertyMatch.group("frame"    )
            component      =       propertyMatch.group("component")
            if (frame     != "rest" and frame     != "observed"):
                raise RuntimeError(funcname+"(): frame must be 'rest' or 'observed'")
            if (component != "disk" and component != "spheroid"):
                raise RuntimeError(funcname+"(): component must be 'disk' or 'spheroid'")
            # Determine the unextincted luminosity name.
            unattenuatedDatasetName = propertyName.replace(":dustCompendium","")
            # Determine extrapolation options.
            extrapolateOpticalDepth = rcParams.getboolean("dustCompendium","extrapolateOpticalDepth",fallback=True)
            # Load dust attenuations.
            if not self.tablesLoaded:
                self.tablesLoaded = True
                attenuationsFileName            = self.data.search(rcParams.get("dustCompendium","attenuationsFile",fallback="compendiumAttenuations.hdf5"))
                attenuationsFile                = h5py.File(attenuationsFileName,'r')
                self.wavelengthTable            = np.array(attenuationsFile['wavelength'                       ])
                self.inclinationTable           = np.array(attenuationsFile['inclination'                      ])
                self.opticalDepthTable          = np.array(attenuationsFile['opticalDepth'                     ])
                self.spheroidScaleRadialTable   = np.array(attenuationsFile['spheroidScaleRadial'              ])
                self.attenuationDiskTable       = np.array(attenuationsFile['attenuationDisk'                  ])
                self.attenuationSpheroidTable   = np.array(attenuationsFile['attenuationSpheroid'              ])
                self.extrapolationDiskTable     = np.array(attenuationsFile['extrapolationCoefficientsDisk'    ])
                self.extrapolationSpheroidTable = np.array(attenuationsFile['extrapolationCoefficientsSpheroid'])
                self.opacity                    = attenuationsFile.attrs.get('opacity')
            # Load filter.
            filters             = GalacticusFilter()
            filter              = filters.load(filterName)
            wavelengthEffective = filter.effectiveWavelength/1.0e4
            # For observed frame luminosities, blueshift the filter to the rest-frame of the galaxy.
            if (frame == "observed"):
                wavelengthEffective /= 1.0+redshift
            # Get all required properties.
            properties = self.galaxies.get(redshift,properties=[ "diskAbundancesGasMetals", "diskRadius", "spheroidRadius", "inclination", unattenuatedDatasetName ])
        # Identify galaxies with viable disks.
        if (component == "spheroid"):
            viable = np.logical_and(properties['diskAbundancesGasMetals'].data > 0.0,properties['spheroidRadius'].data > 0.0)
        else:
            viable =                properties['diskAbundancesGasMetals'].data > 0.0
        # Validate disks.
        if (np.any(properties['diskRadius'].data[viable] <= 0.0)):
            raise RuntimeError(funcname+"(): non-positive disk radius found for disk with positive gas mass")
        # Find the column density of metals through the center of disks [Msolar/Mpc^2].
        columnDensityMetals         = np.zeros(properties['diskAbundancesGasMetals'].data.shape)
        columnDensityMetals[viable] = properties['diskAbundancesGasMetals'].data[viable]/2.0/np.pi/properties['diskRadius'].data[viable]**2
        # Get the dust-to-metals ratio. If noe is provided we use a default of 0.44 which is approximately correct for the Milky
        # Way (e.g. Popping et al.; 2017; http://adsabs.harvard.edu/abs/2017MNRAS.471.3152P).
        dustToMetalsRatio   = rcParams.getboolean("dustCompendium","dustToMetalsRatio",fallback=0.44)
        # Find column density of dust through center of disks [Msolar/Mpc^2].
        columnDensityDust   = columnDensityMetals*dustToMetalsRatio
        # Construct central V-band optical depths. Decompose to dimensionless units, and extract the values.
        opticalDepth        = columnDensityDust*self.opacity*(massSolar/milli)*(centi/megaParsec)**2
        # If the requested dataset was the central optical depth we can return that now. Otherwise continue with the dust
        # attenuation calculation.
        if (propertyName == "tauV0:dustCompendium"):
            tauV0       = Dataset(name=propertyName)
            tauV0.data  = opticalDepth
            return tauV0        
        # Construct spheroid to disk radius ratios.
        if (component == "spheroid"):
            spheroidScaleRadial         = np.ones(properties['spheroidRadius'].data.shape)
            spheroidScaleRadial[viable] = properties['spheroidRadius'].data[viable]/properties['diskRadius'].data[viable]
        # Set wavelengths and inclinations.
        inclination = properties['inclination'].data
        wavelength  = np.ones(inclination.shape)*wavelengthEffective
        # Validate property ranges.
        if (any(inclination <  0.0)):
            raise RuntimeError(funcname+"(): galaxies with inclination <  0 present - this is not permitted")
        if (any(inclination > 90.0)):
            raise RuntimeError(funcname+"(): galaxies with inclination > 90 present - this is not permitted")
        if (any(inclination  < self.inclinationTable [ 0])):
            raise RuntimeError(funcname+"(): galaxies with inclination < "  +str(self.inclinationTable [ 0])+" present - out of range")
        if (any(inclination  > self.inclinationTable [-1])):
            raise RuntimeError(funcname+"(): galaxies with inclination > "  +str(self.inclinationTable [-1])+" present - out of range")
        if (any(wavelength   < self.wavelengthTable  [ 0])):
            raise RuntimeError(funcname+"(): galaxies with wavelength < "   +str(self.wavelengthTable  [ 0])+" present - out of range")
        if (any(wavelength   > self.wavelengthTable  [-1])):
            raise RuntimeError(funcname+"(): galaxies with wavelength > "   +str(self.wavelengthTable  [-1])+" present - out of range")
        if (any(opticalDepth < self.opticalDepthTable[ 0])):
            raise RuntimeError(funcname+"(): galaxies with optical depth < "+str(self.opticalDepthTable[ 0])+" present - out of range")
        if (any(opticalDepth > self.opticalDepthTable[-1]) and not extrapolateOpticalDepth):
            raise RuntimeError(funcname+"(): galaxies with optical depth > "+str(self.opticalDepthTable[-1])+" present - out of range")
        if (component == "spheroid"):
            if (any(spheroidScaleRadial < self.spheroidScaleRadialTable[ 0])):
                raise RuntimeError(funcname+"(): galaxies with spheroid radial scale < "+str(self.spheroidScaleRadialTable[ 0])+" present - out of range")
            if (any(spheroidScaleRadial > self.spheroidScaleRadialTable[-1])):
                raise RuntimeError(funcname+"(): galaxies with spheroid radial scale > "+str(self.spheroidScaleRadialTable[-1])+" present - out of range")
        # Determine where extrapolation is needed.
        if (extrapolateOpticalDepth):
            interpolated=opticalDepth <= self.opticalDepthTable[-1]
            extrapolated=opticalDepth >  self.opticalDepthTable[-1]
        else:
            interpolated=numpy.full(opticalDepth.shape,True ,dtype=bool)
            extrapolated=numpy.full(opticalDepth.shape,False,dtype=bool)
        # Build interpolator and extrapolator.
        if (component == "disk"):
            # Build interpolator and interpolants for disk component.
            interpolator       = scipy.interpolate.RegularGridInterpolator((self.wavelengthTable,self.inclinationTable,self.opticalDepthTable                              ),self.attenuationDiskTable              )
            galaxyInterpolants = np.transpose(np.stack((wavelength,inclination,opticalDepth                    )))
            extrapolator0      = scipy.interpolate.RegularGridInterpolator((self.wavelengthTable,self.inclinationTable                                                     ),self.extrapolationDiskTable   [0,:,:  ])
            extrapolator1      = scipy.interpolate.RegularGridInterpolator((self.wavelengthTable,self.inclinationTable                                                     ),self.extrapolationDiskTable   [1,:,:  ])
            galaxyExtrapolants = np.transpose(np.stack((wavelength,inclination                                 )))
        else:
            # Build interpolator and interpolants for spheroid component.
            interpolator       = scipy.interpolate.RegularGridInterpolator((self.wavelengthTable,self.inclinationTable,self.opticalDepthTable,self.spheroidScaleRadialTable),self.attenuationSpheroidTable           )
            galaxyInterpolants = np.transpose(np.stack((wavelength,inclination,opticalDepth,spheroidScaleRadial)))
            extrapolator0      = scipy.interpolate.RegularGridInterpolator((self.wavelengthTable,self.inclinationTable                       ,self.spheroidScaleRadialTable),self.extrapolationSpheroidTable[0,:,:,:])
            extrapolator1      = scipy.interpolate.RegularGridInterpolator((self.wavelengthTable,self.inclinationTable                       ,self.spheroidScaleRadialTable),self.extrapolationSpheroidTable[1,:,:,:])
            galaxyExtrapolants = np.transpose(np.stack((wavelength,inclination             ,spheroidScaleRadial)))
        # Perform the interpolation.
        attenuations                                      = np.ones(properties[unattenuatedDatasetName].data.shape)
        attenuations[np.logical_and(viable,interpolated)] = interpolator(galaxyInterpolants[np.logical_and(viable,interpolated)])
        # Perform the extrapolations.
        attenuations[np.logical_and(viable,extrapolated)] = np.exp(extrapolator0(galaxyExtrapolants[np.logical_and(viable,extrapolated)])+extrapolator1(galaxyExtrapolants[np.logical_and(viable,extrapolated)])*np.log(opticalDepth[np.logical_and(viable,extrapolated)]))
        # Construct the attenuation luminosity.
        luminosityExtinguished       = Dataset(name=propertyName)
        luminosityExtinguished.attr  = properties[unattenuatedDatasetName].attr
        luminosityExtinguished.data  = properties[unattenuatedDatasetName].data
        luminosityExtinguished.data *= attenuations
        return luminosityExtinguished
