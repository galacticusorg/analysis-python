#! /usr/bin/env python

import sys,os,re,fnmatch
import numpy as np
from ... import rcParams
from ...constants import angstrom,micron
from .manager import ScreenLaw

@ScreenLaw.register_subclass('Prevot')
class Prevot(ScreenLaw):

    def __init__(self):
        self.attrs = {}
        self.attrs["Rv"] = rcParams.getfloat("dustPrevot","Rv",fallback=3.1)
        wavelengths = np.array([1275., 1330., 1385., 1435., 1490., 1545., 1595., 1647., 1700.,\
                                    1755., 1810., 1860., 1910., 2000., 2115., 2220., 2335., 2445.,\
                                    2550., 2665., 2778., 2890., 2995., 3105., 3704., 4255., 5291.,\
                                    12500., 16500., 22000.])
        wavelengths *= angstrom/micron
        klambda = np.array([13.54, 12.52, 11.51, 10.80, 9.84, 9.28, 9.06, 8.49, 8.01, 7.71, 7.17, \
                                6.90, 6.76, 6.38, 5.85, 5.30, 4.53, 4.24, 3.91, 3.49, 3.15, 3.00, \
                                2.65, 2.29, 1.81, 1.00, 0.00, -2.02, -2.36, -2.47]) 
        klambda += Rv
        klambda /= Rv
        self.curve = interp1d(wavelengths,klambda,kind='linear',fill_value="extrapolate")
        return
