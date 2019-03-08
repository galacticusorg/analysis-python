#!/usr/bin/env python
import sys
import re
import numpy as np
from galacticus.io import GalacticusHDF5
from galacticus.galaxies import Galaxies
from galacticus.datasets import Dataset
from galacticus.properties import Property
from galacticus.Cloudy import CloudyTable


# Compute (and store) emission luminosities for all emission lines available in a Galacticus output file.
# -- Alex Merson (05-March-2019)

# Initialize CloudyTable class
CLOUDY = CloudyTable()
# List all available emission lines
emlines = CLOUDY.listAvailableLines()

# Get the file name from the command line.
if ( len(sys.argv) != 2 ):
    sys.exit("Usage: addEmissionLines.py <galacticusModelFile>")
galacticusFileName = sys.argv[1]

print("Adding luminosities for emission lines: "+", ".join(emlines))

# We simply compute the emission lines for each output of the Galacticus output file.
model          = GalacticusHDF5(galacticusFileName,'r+')
galaxies       = Galaxies(model)

for redshift in model.availableRedshifts():
    print("Processing output at z="+str(redshift))
    # Check that this output contains galaxies
    numberGalaxies = model.countGalaxiesAtRedshift(redshift)
    # If output does not contain any galaxies then skip this output
    if numberGalaxies == 0:
        continue
    # Get redshift string
    zStr = model.getRedshiftString(redshift) # e.g. z1.0000
    # Build a list of emission line luminosities.    
    propertyNames = ["totalLineLuminosity:"+name+":observed:"+zStr for name in emlines]
    for propertyName in propertyNames:
        print("   Computing emission line luminosity: "+propertyName)
        property = galaxies.retrieveProperty(propertyName,redshift)
        #model.writeGalacticusDataset(redshift,property)
model.close()
