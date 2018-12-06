#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from .. import rcParams
from ..properties.manager import Property
from ..datasets import Dataset
from ..Cloudy import CloudyTable
from ..constants import kilo,angstrom
from ..constants import speedOfLight
from ..constants import Pi


@Property.register_subclass('fwhm')
class FullWidthHalfMaximum(Property):
    
    def  __init__(self,galaxies):
        self.galaxies = galaxies
        self.CLOUDY = CloudyTable()
        return

    def get(self,propertyName,redshift):
        """
        FullWidthHalfMaximum.get(): Compute full width half maximum for an emission line.

        USAGE:  DATA = FullWidthHalfMaximum.get(propertyName,redshift)

           INPUTS
                propertyName -- Name of FWHM property to compute.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUT
                DATA         -- Instance of galacticus.datasets.Dataset() class containing
                                computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        MATCH = self.parseDatasetName(propertyName)
        # Get rest wavelength of line
        restWavelength = self.CLOUDY.getWavelength(MATCH.group('lineName'))
        # Get line width velocity
        widthVelocity = self.getVelocityWidth(propertyName,redshift)
        # Create dataset
        DATA = Dataset(name=propertyName)
        attr = {"unitsInSI":angstrom}
        DATA.attr = attr
        # Compute and store FWHM in Angstroms
        c = speedOfLight/kilo
        DATA.data = restWavelength*(widthVelocity/c)
        return DATA

    def getApproximateVelocityDispersion(self,redshift):
        """
        FullWidthHalfMaximum.getApproximateVelocityDispersion(): Compute approximate velocity dispersion.
        

        USAGE  velocity = FullWidthHalfMaximum.getApproximateVelocityDispersion(redshift)
        
           INPUTS
              redshift   -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUTS
              velocity   -- Approximate velocity dispersion in km/s.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get parameters for velocity dispersion
        scaleVelocityRatio = rcParams.getfloat("velocityDispersion","scaleVelocityRatio")
        minVelocityDipserion = rcParams.getfloat("velocityDispersion","minVelocityDipserion")        
        # Read necessary galaxy properties
        properties = ["spheroidVelocity","diskVelocity","inclination"]
        GALS = self.galaxies.get(redshift,properties)
        # Approximate spheroid velocity dispersion using spheroid 'rotation velocity'
        approximateVelocityDispersion = np.copy(GALS["spheroidVelocity"].data)
        # Determine spheroid-to-total mass ratio
        baryonicSpheroidToTotalRatio = self.getBaryonicBulgeToTotalRatio(redshift)
        # For any empty halos not removed by mask, set velocity dispersion to specified minium value
        emptyHalos = 999.9
        mask = baryonicSpheroidToTotalRatio == emptyHalos
        np.place(approximateVelocityDispersion,mask,minVelocityDipserion)
        # Check if any disk-dominated galaxies in dataset and replace corresponding velocities
        diskDominated = baryonicSpheroidToTotalRatio<0.5
        if any(diskDominated):
            # Approximate disk velocity dispersion using combiantion
            # of disk rotational velocity and disk vertical velocity
            # (computed as fraction of rotation velocity)
            diskVelocity = np.copy(GALS["diskVelocity"].data)
            inclination = np.copy(GALS["inclination"].data)
            degrees = rcParams.getboolean("inclination","degrees")
            if degrees:
                inclination *= (Pi/180.0)
            diskVelocity *= np.sqrt(np.sin(inclination)**2+(scaleVelocityRatio*np.cos(inclination))**2)
            np.place(approximateVelocityDispersion,diskDominated,diskVelocity[diskDominated])
        return approximateVelocityDispersion

    def getBaryonicBulgeToTotalRatio(self,redshift):
        """
        FullWidthHalfMaximum.getBaryonicBulgeToTotalRatio(): Compute the bulge-to-total ratio for baryonic matter
                                                             (stellar mass + cold gas) in galaxies.
        
        USAGE: ratio = FullWidthHalfMaximum.getBaryonicBulgeToTotalRatio(redshift)
        
           INPUTS
               redshift -- Redshift value to query Galacticus HDF5 outputs. 

           OUTPUTS
               ratio    -- Bulge-to-total ratio for baryons (stellar mass + cold gas).

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        emptyHalos = 999.9
        # Determine spheroid-to-total mass ratio          
        properties = ["diskMassStellar","spheroidMassStellar",
                      "diskMassGas","spheroidMassGas"]
        GALS = self.galaxies.get(redshift,properties=properties)        
        baryonicSpheroidMass = np.copy(GALS["spheroidMassStellar"].data+GALS["spheroidMassGas"].data)
        baryonicDiskMass = np.copy(GALS["diskMassStellar"].data+GALS["diskMassGas"].data)
        totalBaryonicMass = baryonicSpheroidMass + baryonicDiskMass
        mask = totalBaryonicMass == 0.0
        np.place(totalBaryonicMass,mask,1.0)
        np.place(baryonicSpheroidMass,mask,emptyHalos)
        return baryonicSpheroidMass/totalBaryonicMass

    def getVelocityWidth(self,propertyName,redshift):
        """
        FullWidthHalfMaximum.getVelocityWidth(): Estimate the velocity width of an emission line either
                                                 assuming a ficed wdith or by approximation from the 
                                                 velocity dispersion of the galaxy.

        USAGE: velocity = FullWidthHalfMaximum.getVelocityWidth(propertyName,redshift)

            INPUTS
                propertyName -- Name of FWHM property to compute.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.

            OUTPUTS
                velocity     -- Velocity width in km/s.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if fnmatch.fnmatch(MATCH.group('width'),"fixedWidth*"):
            fixedWidth = float(MATCH.group('width').replace("fixedWidth",""))
            ngals = self.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
            width = np.ones(ngals,dtype=float)*fixedWidth
        elif fnmatch.fnmatch(MATCH.group('width'),"dispersionWidth"):
            width = self.getApproximateVelocityDispersion(redshift)
        else:
            msg = funcname+"(): line width method must be "+\
                "'fixedWidth<velocity>' or 'dispersionWidth'!"
            raise ValueError(msg)
        return width

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        FullWidthHalfMaximum.matches(): Returns boolean to indicate whether this class can 
                                        process the specified property.

        USAGE: match =  FullWidthHalfMaximum.matches(propertyName,[redshift=None])

          INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.

          OUTPUTS
              match        -- Boolean indicating whether this class can process
                              this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid full width half maximum dataset."
            raise RuntimeError(msg)
        return False

    def parseDatasetName(self,datasetName):
        """
        FullWidthHalfMaximum.parseDatasetName: Parse a FWHM dataset name.

        USAGE: SEARCH = FullWidthHalfMaximum.parseDatasetName(propertyName)

             INPUTS 
                propertyName -- Property name to parse.

             OUTPUTS 
                SEARCH       -- Regex seearch (re.search) object or None 
                                if propertyName cannot be parsed.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Construct search string to pass to regex
        lines = "(?P<lineName>"+"|".join(self.CLOUDY.listAvailableLines())+")"
        velocityStr = "(?P<width>dispersionWidth|fixedWidth[\d\.]+)"
        searchString = "^fullWidthHalfMaximum:"+lines+":"+velocityStr+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            "(?P<recent>:recent)?$"
        return re.search(searchString,datasetName)
    




        
        

