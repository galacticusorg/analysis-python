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

    def retrieveProperty(self,propertyName,redshift):
        propertyDataset = None
        for property,propertyClass in self.Property.subclasses.items():
            #print "Testing for match on "+property
            PC = propertyClass(self)
            if (PC.matches(propertyName,redshift=redshift)):
                # We have a class that matches our property.                                                                                           
                print "   Class "+property+" matches"
                propertyDataset = PC.get(propertyName,redshift)
        return propertyDataset


    def get(self,z,properties=None):
        # Store galaxy properties and store information in dictionary
        GALAXIES = {propertyName:self.retrieveProperty(propertyName,z) \
                        for propertyName in properties}
        return GALAXIES

        
        
