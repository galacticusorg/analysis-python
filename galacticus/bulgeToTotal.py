#! /usr/bin/env python

import sys,os,re
import numpy as np
import unittest
import warnings
from .datasets import Dataset
from .properties.manager import Property


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

    def parseDatasetName(self,propertyName):
        """
        BulgeToTotal.parseDatasetName(): Parse a bulge-to-total dataset name.

        USAGE: MATCH = BulgeToTotal.parseDatasetName(propertyName)

           INPUTS 
              propertyName -- Property name to parse.
         
           OUTPUTS 
                    MATCH  -- Regex search (re.search) object or None
                              if propertyName cannot be parsed.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        searchString = "^bulgeToTotal(?P<phase>\w+)$"
        MATCH = re.search(searchString,propertyName)
        return MATCH


    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        BulgeToTotal.matches(): Returns boolean to indicate whether this class can 
                                process the specified property.

        USAGE: match =  BulgeToTotal.matches(propertyName,[redshift=None])

          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs. 
              raiseError   -- Raise exception if propertyName does not match 
                              (Default=False).

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
                "' is not a valid bulge-to-total property."
            raise RuntimeError(msg)
        return False
    
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
        assert(self.matches(propertyName,redshift=redshift,raiseError=True))
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
