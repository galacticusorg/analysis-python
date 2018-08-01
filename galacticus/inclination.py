#! /usr/bin/env python

import numpy as np
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
    
    def __init__(self):
        self.datatype = float

    def matches(self,propertyName):
        return propertyName == "inclination"

    def compute(self,propertyName,redshift,galaxies,degrees=True):        
        N = galaxies.GH5Obj.countGalaxies(redshift)
        inclination = Generate_Random_Inclinations(N,degrees=degrees)
        inclination = np.random.random(10)
        return inclination
    
