#! /usr/bin/env python

import __future__

from ..filters.filters import GalacticusFilter
from ..Cloudy import CloudyTable

CLOUDY = CloudyTable()
FILTERS = GalacticusFilter()

def getEffectiveWavelength(regexMatch,redshift):    
    # Identify whether luminosity is an emission line or a stellar luminosity                                                                                               
    if regexMatch.group('filterName') is not None:
        FILTER = FILTERS.load(regexMatch.group('filterName').replace(":",""))
        wavelength = np.ones_like(redshift)*float(FILTER.effectiveWavelength)
        if regexMatch.group('frame') == "observed":
            wavelength /= (1.0+redshift)            
    else:
        lineName = regexMatch.group("lineName")
        wavelength = np.ones_like(redshift)*float(CLOUDY.getWavelength(lineName))
    return wavelength
