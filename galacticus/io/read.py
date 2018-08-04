#! /usr/bin/env python

from ..properties.manager import Property

@Property.register_subclass('readhdf5')
class ReadHDF5(Property):
    """
    Read: Manage reading of galaxy data from HDF5 file.

    Functions: 
          matches(): Indicates whether specified dataset can be
                     processed by this class.  
          get(): Extracts galaxy property at specified redshift.

    """
    def __init__(self,galaxies):
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None):
        """
        Read.matches(): Returns boolean to indicate whether this class can 
                        process the specified property.

        USAGE: match = Read.matches(propertyName,[redshift=None])

          INPUTS 
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.

          OUTPUTS 
              match        -- Boolean indicating whether this class can 
                              process this property.

        """
        return self.galaxies.GH5Obj.datasetExists(propertyName,redshift)

    def get(self,propertyName,redshift=None):
        """
        Read.get(): Extract galaxy property for specified redshift.

        USAGE:  DATA = Inclination.get(propertyName,redshift)

          INPUTS 
             propertyName -- Name of property to extract. 
             redshift     -- Redshift value to query Galacticus HDF5 outputs.

           OUTPUT 
             DATA         -- Instance of galacticus.datasets.Dataset()
                             class containing computed galaxy information.

        """
        return self.galaxies.GH5Obj.getDataset(propertyName,redshift)
