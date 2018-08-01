#! /usr/bin/env python

from .properties.manager import Property

class Galaxies(object):
    
    def __init__(self,GH5Obj=None,verbose=True):        
        self.GH5Obj = GH5Obj
        self.verbose = verbose
        self.Property = Property()
        return

    def updateGH5Obj(self,GH5Obj):
        """
        Updates the GalacticusHDF5 object instance (so can read from another file).
        """
        self.GH5Obj = GH5Obj
        return

    def retrieveProperty(self,propertyName):
        for property,propertyClass in self.Property.subclasses.items():
            print "Testing for match on "+property
            PC = propertyClass()
            if (PC.matches(propertyName)):
                # We have a class that matches our property.                                                                                           
                print "   Class "+property+" matches"
                propertyValues = PC.compute(propertyName,self)
                print propertyValues        
        return


    def get(self,z,properties=None):
        # Create numpy array to store galaxy properties
        # ...
        # Store galaxy properties and store in numpy array
        dummy = [self.retrieveProperty(propertyName) for propertyName in properties]
        return

        
        
