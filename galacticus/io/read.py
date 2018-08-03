#! /usr/bin/env python

from ..properties.manager import Property

@Property.register_subclass('readhdf5')
class ReadHDF5(Property):

    def __init__(self,galaxies):
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None):
        return self.galaxies.GH5Obj.datasetExists(propertyName,redshift)

    def get(self,propertyName,redshift=None):
        return self.galaxies.GH5Obj.getDataset(propertyName,redshift)
