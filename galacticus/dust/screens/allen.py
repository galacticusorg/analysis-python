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

class UnitTest(unittest.TestCase):

    def test(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Allen: "+funcname)
        print("Testing Allen class")
        rcParams.update("dustAllen","Rv",3.09)
        DUST = Allen()
        self.assertEqual(DUST.attrs["Rv"],3.09)
        self.assertIsNotNone(DUST.curve)
        self.assertTrue(type(DUST.curve(0.01)),float)
        self.assertTrue(type(DUST.curve(0.1)),float)
        self.assertTrue(type(DUST.curve(0.2)),float)
        self.assertTrue(type(DUST.curve(0.3)),float)
        self.assertTrue(type(DUST.curve(1.0)),float)
        print("TEST COMPLETE")
        print("\n")
        return
