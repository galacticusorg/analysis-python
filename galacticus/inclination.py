#! /usr/bin/env python

import sys
import numpy as np
from . import rcParams
from .datasets import Dataset
from .properties.manager import Property
from .constants import Pi

def Generate_Random_Inclinations(N,degrees=True):
    """ 
    Generate_Random_Inclinations: Return a list of N random inclination angles.

    USAGE: inc = Generate_Random_Inclinations(N,[degrees]) 
    
     INTPUTS
        N       : Integer number of angles to generate.                                                                                              
        degrees : Return angles in degrees? (Default value = True)                                                                                   
    
     OUTPUTS    
        inc     : Numpy array of N inclination angles.                                                                                               
    
    """
    angles = np.arccos(np.random.rand(N))
    if degrees:
        angles *= 180.0/Pi
    return angles

@Property.register_subclass('inclination')
class Inclination(Property):
    """
    Inclination: Compute galaxy inclinations.

    Functions:
            matches(): Indicates whether specified dataset can be processed by this class.
            compute(): Computes galaxy inclinations at specified redshift.

    Attributes:
           datatype: float

    """    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.datatype = float
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None):
        """
        Inclination.matches(): Returns boolean to indicate whether this class can process
                               the specified preperty.

        USAGE: match =  Inclination.matches(propertyName,[redshift=None])

          INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. (Redundant
                              in this particular case, but required for other properties.)                              

          OUTPUTS
              match        -- Boolean indicating whether this class can process 
                              this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return propertyName == "inclination"

    def get(self,propertyName,redshift):        
        """
        Inclination.compute(): Compute galaxy inclinations for specified redshift.
        
        USAGE:  DATA = Inclination.compute(propertyName,redshift,galaxies)
                
           INTPUTS
           
                propertyName -- Name of property to compute. This should be set to 'inclination'.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.
                galaxies     -- Instance of galacticus.galaxies.Galaxies() class.
           
           OUTPUT
                DATA         -- Instance of galacticus.datasets.Dataset() class containing 
                                computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            raise RunimeError(funcname+"(): Specified propert '"+propertyName+"' is not an inclination.")
        degrees = rcParams.getboolean("inclination","degrees")
        N = self.galaxies.GH5Obj.countGalaxies(redshift)
        inclination = Generate_Random_Inclinations(N,degrees=degrees)
        inclination = np.random.random(N)
        DATA = Dataset(name="inclination",data=inclination,attr={"degrees":degrees})
        return DATA
    
