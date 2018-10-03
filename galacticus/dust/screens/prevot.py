#! /usr/bin/env python

import sys,os,re,fnmatch
import numpy as np
import unittest
from scipy.interpolate import interp1d
from ... import rcParams
from ...constants import angstrom,micron
from .manager import ScreenLaw
from .utils import getPrevotDustTable

@ScreenLaw.register_subclass('Prevot')
class Prevot(ScreenLaw):

    def __init__(self):
        self.attrs = {}
        self.attrs["Rv"] = rcParams.getfloat("dustPrevot","Rv",fallback=3.1)
        table = getPrevotDustTable()
        table.klambda += self.attrs["Rv"]
        table.klambda /= self.attrs["Rv"]
        self.curve = interp1d(table.wavelength,table.klambda,\
                                  kind='linear',fill_value="extrapolate")
        return

