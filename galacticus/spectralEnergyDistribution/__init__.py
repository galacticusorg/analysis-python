#! /usr/bin/env python

import re
import __future__
import numpy as np

def parseDatasetName(datasetName):
    # Construct search string to pass to regex                                                                                                                              
    searchString = "^(?P<component>disk|spheroid|total)SpectralEnergyDistribution:"
    lower = "(?P<lowerWavelength>[\d\.]+)"
    upper = "(?P<upperWavelength>[\d\.]+)"
    resolution = "(?P<resolution>[\d\.]+)"
    wavelengths = "(?P<wavelengths>"+lower+"_"+upper+"_"+resolution+")"
    searchString = searchString + wavelengths + ":(?P<frame>rest|observed)"+\
                   "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
                   "(?P<lineWidth>:fixedWidth[\d\.]+)?"+\
                   "(?P<snrString>:snr(?P<snr>[\d\.]+))?"+\
                   "(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"+\
                   "(?P<noLines>:noLines)?"
    return re.search(searchString,datasetName)

def getSpectralEnergyDistributionWavelengths(datasetName):
    MATCH = parseDatasetName(datasetName)
    lowerWavelength = float(MATCH.group("lowerWavelength"))
    upperWavelength = float(MATCH.group("upperWavelength"))
    resolution = float(MATCH.group("resolution"))
    wavelengths = np.arange(lowerWavelength,upperWavelength+resolution,resolution)
    return wavelengths
