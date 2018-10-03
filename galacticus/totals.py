#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
import unittest
from .datasets import Dataset
from .properties.manager import Property

@Property.register_subclass('totals')
class Totals(Property):
    """
    Totals: Compute a total property by summing up the disk and spheroid components.

    Functions: 
            matches(): Indicates whether specified dataset can be processed by this class.  
            get(): Computes galaxy total at specified redshift.

    """    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return


    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        Totals.matches(): Returns boolean to indicate whether this class can 
                          process the specified property.

        uSAGE: match =  Totals.matches(propertyName,[redshift=None])
        
          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. 
              raiseError   -- Raise exception if unable to match property name.

          OUTPUTS 
              match        -- Boolean indicating whether this class can process 
                              this property.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        match = propertyName.startswith("total")
        exemptOptions = ["totalMetallicity","totalMagnitude*"]
        exempt = [fnmatch.fnmatch(propertyName,option) for option in exemptOptions]
        match = match and not any(exempt)
        if not match and raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid totals property."
            raise RuntimeError(msg)
        return match
    
    def get(self,propertyName,redshift):
        """
        Totals.get(): Compute total galaxy property using disk and spheroid 
                      components at specified redshift.

        USAGE: DATA = Totals.get(propertyName,redshift)

           INPUTS
             propertyName -- Name of total property to compute. This name
                             shoud start with 'total'.
             redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUTS
              DATA        -- Instance of galacticus.datasets.Dataset() class 
                             containing computed galaxy information, or None
                             if one of the components is missing.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Get disk and spheroid properties
        components = [propertyName.replace("total","disk"),propertyName.replace("total","spheroid")]
        GALS = self.galaxies.get(redshift,properties=components)
        if any([GALS[key] is None for key in GALS.keys()]):
            return None
        # Sum components and return total
        DATA = Dataset(name=propertyName)
        DATA.attr = GALS[components[0]].attr
        DATA.data = np.copy(GALS[components[0]].data+GALS[components[1]].data)
        del GALS
        return DATA

