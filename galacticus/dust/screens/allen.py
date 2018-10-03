#! /usr/bin/env python

import sys,os,re,fnmatch
import numpy as np
import unittest
from scipy.interpolate import interp1d
from ... import rcParams
from .manager import ScreenLaw
from .utils import getAllenDustTable

@ScreenLaw.register_subclass('Allen')
class Allen(ScreenLaw):
    
    def __init__(self):
        self.attrs = {}
        self.attrs["Rv"] = rcParams.getfloat("dustAllen","Rv",fallback=3.1)
        table = getAllenDustTable()
        self.curve = interp1d(table.wavelength,table.klambda,\
                                  kind='linear',fill_value="extrapolate")
        return
