#! /usr/bin/env python

import sys
import numpy as np
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
        
    def matches(self,propertyName,redshift=None):
        """
        Totals.matches(): Returns boolean to indicate whether this class can 
                          process the specified preperty.

        USAGE: match =  Totals.matches(propertyName,[redshift=None])                                                                                                       
        
          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. 

          OUTPUTS 
              match        -- Boolean indicating whether this class can process 
                              this property.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return propertyName.startswith("total")
    
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
        if not self.matches(propertyName):
            raise RunimeError(funcname+"(): Cannot process property '"+propertyName+"'.")
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


@Property.register_subclass('bulgetototal')
class BulgeToTotal(Property):
    """
    BulgeToTotal: Compute a bulge-to-total ratio based upon specified galaxy property.

    Functions: 
            matches(): Indicates whether specified dataset can be processed by this class.  
            get(): Computes bulge-to-total ratio at specified redshift.

    """    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None):
        """
        BulgeToTotal.matches(): Returns boolean to indicate whether this class can 
                                process the specified preperty.

        USAGE: match =  BulgeToTotal.matches(propertyName,[redshift=None])

          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. 

          OUTPUTS 
              match        -- Boolean indicating whether this class can process 
                              this property.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return propertyName.startswith("bulgeToTotal")
    
    def get(self,propertyName,redshift):
        """
        BulgeToTotal.get(): Compute bulge-to-total ratio for specified galaxy property 
                            using total and spheroid components at specified redshift.

        USAGE: DATA = BulgeToTotal.get(propertyName,redshift)

           INPUTS
             propertyName -- Name of bulge-to-total ratio to compute. This name
                             shoud start with 'bulgeToTotal'.
             redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUTS
              DATA        -- Instance of galacticus.datasets.Dataset() class 
                             containing computed galaxy information, or None
                             if one of the components is missing.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.matches(propertyName):
            raise RunimeError(funcname+"(): Cannot process property '"+propertyName+"'.")
        # Get spheroid and total properties
        spheroid = propertyName.replace("bulgeToTotal","spheroid")
        total = propertyName.replace("bulgeToTotal","total")
        GALS = self.galaxies.get(redshift,properties=[spheroid,total])
        if any([GALS[key] is None for key in GALS.keys()]):
            return None
        # Compute ratio and return result
        DATA = Dataset(name=propertyName)
        DATA.attr = {}
        DATA.data = np.copy(GALS[spheroid].data/GALS[total].data)
        del GALS
        return DATA

