#! /usr/bin/env python 

import sys,os,re,fnmatch
import numpy as np
import unittest
from scipy.interpolate import interp1d
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
        wavelengths = np.linspace(0.12,0.333,2200)
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

    
class UnitTest(unittest.TestCase):

    def test(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Fitzpatrick: "+funcname)
        print("Testing Fitzpatrick class")
        rcParams.update("dustFitzpatrick","Rv",2.71)
        DUST = Fitzpatrick()
        self.assertEqual(DUST.attrs["Rv"],2.71)
        self.assertIsNotNone(DUST.curve)
        wavelengths = np.array([0.01,0.12,1.0,10.0,20.0])
        self.assertTrue(type(DUST.curve(wavelengths)),np.ndarray)
        [self.assertTrue(type(DUST.curve(w)),float) for w in wavelengths]
        print("TEST COMPLETE")
        print("\n")
        return
