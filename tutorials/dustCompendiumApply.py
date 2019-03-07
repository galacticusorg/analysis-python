#!/usr/bin/env python
import sys
import re
import numpy as np
from galacticus.io import GalacticusHDF5
from galacticus.galaxies import Galaxies
from galacticus.datasets import Dataset
from galacticus.properties import Property

# Compute (and store) dust-extinguished luminosities for all stellar luminosity bands in a Galacticus output file by applying the
# "dust compendium" extinction model.
# -- Andrew Benson (08-August-2018)

# Get the file name from the command line.
if ( len(sys.argv) != 2 ):
    sys.exit("Usage: dustCompendiumApply.py <galacticusModelFile>")
galacticusFileName = sys.argv[1]

# We simply retrieve all available outputs and stellar luminosities from the file and then retrieve the equivalent
# dust-extinguished version.
model          = GalacticusHDF5(galacticusFileName,'r+')
galaxies       = Galaxies(model)
for redshift in model.availableRedshifts():
    print "Processing output at z="+str(redshift)+":"
    # Build a list of luminosity properties that we want to apply dust to. The ionizing continuum bands are excluded as usually
    # these are only of interest for emission line calculations.
    propertyNames = list(filter(lambda x: re.search("^(disk|spheroid)LuminositiesStellar:(?!Lyc|HeliumContinuum|OxygenContinuum|SED)[^:]+:(rest|observed):z[\d\.]+$",x),model.availableDatasets(redshift)))
    propertyNames.append('inclination')
    for propertyName in propertyNames:
        if (propertyName == "inclination"):
            propertyNameDusty = propertyName
        else:
            print "   Computing dust extinguished luminosity for band: "+propertyName
            propertyNameDusty = propertyName+":dustCompendium"
        property = galaxies.retrieveProperty(propertyNameDusty,redshift)
        model.writeGalacticusDataset(redshift,property)
