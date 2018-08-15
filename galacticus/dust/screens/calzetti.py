#! /usr/bin/env python

import sys,os,re,fnmatch
import numpy as np
import unittest
from scipy.interpolate import interp1d
from ... import rcParams
from ...constants import angstrom,micron
from .manager import ScreenLaw

@ScreenLaw.register_subclass('Calzetti')
class Calzetti(ScreenLaw):

    def __init__(self):
        self.attrs = {}
        self.attrs["Rv"] = rcParams.getfloat("dustCalzetti","Rv",fallback=4.05)
        self.curve = None
        wavelengths = np.linspace(0.12,2.20,20800)        
        dtype = [("wavelength",float),("klambda",float)]
        dustTable = np.zeros(len(wavelengths),dtype=dtype).view(np.recarray)
        dustTable.wavelength = np.copy(wavelengths)
        lower = 2.659*( -2.156+(1.509/wavelengths)-(0.198/wavelengths**2)+\
                             (0.011/wavelengths**3) )
        upper = 2.659*( -1.857 + (1.040/wavelengths) )
        mask = dustTable.wavelength >= 0.63
        dustTable.klambda = np.copy(lower)
        np.place(dustTable.klambda,mask,np.copy(upper[mask]))
        dustTable.klambda += self.attrs["Rv"]
        dustTable.klambda /= self.attrs["Rv"]
        self.curve = interp1d(dustTable.wavelength,dustTable.klambda,\
                                  kind='linear',fill_value="extrapolate")
        return


class UnitTest(unittest.TestCase):
    
    def test(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Calzetti: "+funcname)
        print("Testing Calzetti class")
        rcParams.update("dustCalzetti","Rv",4.06)
        DUST = Calzetti()
        self.assertEqual(DUST.attrs["Rv"],4.06)
        self.assertIsNotNone(DUST.curve)        
        wavelengths = np.array([0.01,0.12,1.0,2.2,5.0])
        self.assertTrue(type(DUST.curve(wavelengths)),np.ndarray)
        [self.assertTrue(type(DUST.curve(w)),float) for w in wavelengths]
        print("TEST COMPLETE")
        print("\n")
        return
        
