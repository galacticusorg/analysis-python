#! /usr/bin/env python

import sys,os,re,fnmatch
import numpy as np
from ... import rcParams
from .manager import ScreenLaw

@ScreenLaw.register_subclass('Calzetti')
class Calzetti(ScreenLaw):

    def __init__(self):
        self.attrs = {}
        self.attrs["Rv"] = rcParams.getfloat("dustCalzetti","Rv",fallback=4.05)
        diff = 1.0*angstrom/micron
        wavelengths = np.arange(0.12,2.20+diff,diff)
        dtype = [("wavelength",float),("klambda",float)]
        dustTable = np.zeros(len(wavelengths),dtype=dtype).view(np.recarray)
        dustTable.wavelength = np.copy(wavelengths)
        lower = 2.659*( -2.156+(1.509/wavelengths)-(0.198/wavelengths**2)+\
                             (0.011/wavelengths**3) )
        upper = 2.659*( -1.857 + (1.040/wavelengths) )
        mask = dustTable.wavelength >= 0.63
        dustTable.klambda = np.copy(lower)
        np.place(dustTable.klambda,mask,np.copy(upper[mask]))
        dustTable.klambda += Rv
        dustTable.klambda /= Rv
        self.curve = interp1d(dustTable.wavelength,dustTable.klambda,\
                                  kind='linear',fill_value="extrapolate")
        return
