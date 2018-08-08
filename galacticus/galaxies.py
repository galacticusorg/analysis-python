#! /usr/bin/env python

import sys
import warnings
from .properties.manager import Property

class Galaxies(object):
    
    def __init__(self,GH5Obj=None,verbose=True):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.GH5Obj = GH5Obj
        self.verbose = verbose
        self.Property = Property()
        self.properties = {}
        for property,propertyClass in self.Property.subclasses.items():
            self.properties[property] = propertyClass(self)
        return

    def updateGH5Obj(self,GH5Obj):
        """
        Updates the GalacticusHDF5 object instance (so can read from another file).
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.GH5Obj = GH5Obj
        return

    def retrieveProperty(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        propertyDataset = None
        for property,propertyClass in self.properties.items():
            #print "Testing for match on "+property
            if (propertyClass.matches(propertyName,redshift=redshift)):
                # We have a class that matches our property.                                                                                           
                #print "   Class "+property+" matches"
                propertyDataset = propertyClass.get(propertyName,redshift)
                break
        if propertyDataset is None:
            warnings.warn("\n"+funcname+"(): '"+propertyName+"' returned None instance!")
        return propertyDataset


    def get(self,z,properties=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store galaxy properties and store information in dictionary
        GALAXIES = {propertyName:self.retrieveProperty(propertyName,z) \
                        for propertyName in properties}
        return GALAXIES

        
        
