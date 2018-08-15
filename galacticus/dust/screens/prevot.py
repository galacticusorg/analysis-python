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


class UnitTest(unittest.TestCase):

    def test(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Prevot: "+funcname)
        print("Testing Prevot class")
        rcParams.update("dustPrevot","Rv",3.09)
        DUST = Prevot()
        self.assertEqual(DUST.attrs["Rv"],3.09)
        self.assertIsNotNone(DUST.curve)
        self.assertTrue(type(DUST.curve(0.01)),float)
        self.assertTrue(type(DUST.curve(0.1275)),float)
        self.assertTrue(type(DUST.curve(1.0)),float)
        self.assertTrue(type(DUST.curve(2.2)),float)
        self.assertTrue(type(DUST.curve(5.0)),float)
        print("TEST COMPLETE")
        print("\n")
        return
