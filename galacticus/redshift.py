#! /usr/bin/env python

import sys,fnmatch
import numpy as np
import warnings
from .datasets import Dataset
from .constants import speedOfLight
from .properties.manager import Property

@Property.register_subclass('redshift')
class Redshift(Property):

    def __init__(self,galaxies):
        """
        Redshift: Compute galaxy redshifts. This class can either extract the redshift of
                  a Galacticus HDF5 output (snapshot) or compute an obseved redshift if
                  lightcone information is available in the Galacticus HDF5 file.

        Functions: 
               matches(): Indicates whether specified dataset can be processed by this class.  
               get(): Returns galaxy redshifts for specified Galacticus output.
               getObservedRedshift(): Computes the observed redshift (i.e. with peculiar
                                      velocity information included).
               getSnapshotRedshift(): Returns array containing snapshot redshift information.

        """
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.availableOptions = ["snapshotRedshift","observedRedshift","redshift"]
        return

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        Redshift.matches(): Returns boolean to indicate whether this class 
                            can process the specified property.
        
        USAGE: match =  Redshift.matches(propertyName,[redshift=None])

        INPUTS 
            propertyName -- Name of property to process.  
            redshift     -- Redshift value to query Galacticus HDF5 outputs. 
            
        OUTPUTS                                                                                                                                                               
            match        -- Boolean indicating whether this class can
                            process this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        match = propertyName in self.availableOptions
        if raiseError and not match:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid redshift option. Available options: "+\
                ", ".join(["'"+opt+"'" for opt in self.availableOptions])+"."
            raise RuntimeError(msg)
        return match

    def getObservedRedshift(self,redshift):
        """
        Redshift.getObservedRedshift: Computes the observed redshift (i.e. including peculiar
                                      velocities) for the specified Galacticus HDF5 output. If
                                      no lightcone information is included in the HDF5 file,
                                      then the function will return 'None' instance.
        
        USAGE:  DATA = Redshift.getObservedRedshift(redshift)
        
          INPUTS
              redshift -- Redshift value used to query the Galacticus HDF5 outputs.
          
          OUTPUTS
              DATA     -- Instance of galacticus.datasets.Dataset() containing the
                          observed redshift information, or None instance if no
                          lightcone information can be located in the HDF5 file.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.galaxies.GH5Obj.galaxyDatasetExists("lightconeRedshift",redshift):
            warnings.warn(funcname+"(): Unable to compute observed redshift as not a lightcone output.")
            return None
        # Create Dataset instance 
        DATA = Dataset(name="observedRedshift")        
        # Extract necessary lightcone properties
        required = ["lightconeRedshift","lightconePositionX","lightconePositionY","lightconePositionZ",\
                        "lightconeVelocityX","lightconeVelocityY","lightconeVelocityZ"]
        GALS = self.galaxies.get(redshift,properties=required)        
        X = GALS["lightconePositionX"].data
        Y = GALS["lightconePositionY"].data
        Z = GALS["lightconePositionZ"].data
        VX = GALS["lightconeVelocityX"].data
        VY = GALS["lightconeVelocityY"].data
        VZ = GALS["lightconeVelocityZ"].data
        zCos = GALS["lightconeRedshift"].data
        # Compute galaxy radial velocity
        R = np.sqrt(X**2+Y**2+Z**2)
        v_r = (VX*X + VY*Y + VZ*Z)/R
        # Compute and store observed redshift
        c_kms = speedOfLight/1000.0        
        DATA.data = np.copy((1.0+zCos)*(1.0+v_r/c_kms)-1.0)
        # Clear additional variables
        del X,Y,Z,VX,VY,VZ,zCos,R,v_r,GALS
        return DATA

    def getSnapshotRedshift(self,redshift):
        """
        Redshift.getSnapshotRedshift: Extracts an array with values all set to the
                                      redshift of the specified Galacticus output.
        
        USAGE:  DATA = Redshift.getSnapshotRedshift(redshift)
        
          INPUTS
              redshift -- Redshift value used to query the Galacticus HDF5 outputs.
          
          OUTPUTS
              DATA     -- Instance of galacticus.datasets.Dataset() containing the
                          snapshot redshift information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        DATA = Dataset()
        DATA.name = "snapshotRedshift"
        zsnap = self.galaxies.GH5Obj.nearestRedshift(redshift)
        N = self.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        DATA.data = np.ones(N,dtype=float)*zsnap
        return DATA

    def getRedshift(self,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.galaxies.GH5Obj.galaxyDatasetExists("lightconeRedshift",redshift):
            name = "lightconeRedshift"
        else:
            name = "snapshotRedshift"
        GALS = self.galaxies.get(redshift,properties=[name])
        DATA = GALS[name]
        DATA.name = "redshift"
        return DATA
        

    def get(self,propertyName,redshift):
        """        
        Redshift.get(): Compute galaxy redshifts for specified HDF5 output. If
                        Galacticus output contains lightcone information then
                        observed redshifts can be computed.
        
        USAGE:  DATA = Redshift.get(propertyName,redshift)
        
        INPUTS
            propertyName -- Name of property to compute. 
            redshift     -- Redshift value to query Galacticus HDF5 outputs.

        OUTPUT 
            DATA         -- Instance of galacticus.datasets.Dataset() class
                            containing computed galaxy redshift, or None if
                            redshift cannot be computed (e.g. if lightcone
                            information cannot be located in the HDF5 file).

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        if propertyName == "snapshotRedshift":
            return self.getSnapshotRedshift(redshift)
        if propertyName == "observedRedshift":
            return self.getObservedRedshift(redshift)
        if propertyName == "redshift":
            return self.getRedshift(redshift)
        return None
