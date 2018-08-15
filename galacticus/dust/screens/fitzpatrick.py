#! /usr/bin/env python 

import sys,os,re,fnmatch
import numpy as np
from ... import rcParams
from ...constants import angstrom,micron
from .manager import ScreenLaw
from .utils import colorRatio,getAllenDustTable

@ScreenLaw.register_subclass('Fitzpatrick')
class Fitzpatrick(ScreenLaw):

    def __init__(self):
        self.attrs = {}
        self.attrs["Rv"] = rcParams.getfloat("dustFitzpatrick","Rv",fallback=2.72)
        # Get wavelength range
        diff = 1.0*angstrom/micron
        low = 1200.0
        upp = 3330.0
        wavelengths = np.arange(low*angstrom/micron,(upp*angstrom/micron)+diff,diff)
        # Compute dust table using colour ratio
        klambda = colorRatio(wavelengths,"LMC") + self.attrs["Rv"]
        # Build dust table (using Allen et al. dust curve for long
        # wavelengths)
        table = getAllenDustTable()
        mask = table.wavelength > wavelengths.max()
        N = len(wavelengths)+len(table.wavelength[mask])
        dustTable = np.zeros(N,dtype=[("wavelength",float),("klambda",float)]).view(np.recarray)
        dustTable.wavelength = np.append(np.copy(wavelengths),np.copy(table.wavelength[mask]))
        dustTable.klambda = np.append(np.copy(klambda),np.copy(table.klambda[mask]*self.attrs["Rv"]))
        dustTable.klambda /= self.attrs["Rv"]
        self.curve = interp1d(dustTable.wavelength,dustTable.klambda,\
                                  kind='linear',fill_value="extrapolate")
        return

    
