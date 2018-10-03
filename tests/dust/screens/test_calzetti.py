#! /usr/bin/env python

import os,sys
import unittest
import numpy as np
from galacticus.dust.screens.manager import ScreenLaw
from galacticus.dust.screens.calzetti import Calzetti
from galacticus import rcParams

class TestCalzetti(unittest.TestCase):
    
    def test_Calzetti(self):
        rcParams.update("dustCalzetti","Rv",4.06)
        DUST = Calzetti()
        self.assertEqual(DUST.attrs["Rv"],4.06)
        self.assertIsNotNone(DUST.curve)
        wavelengths = np.array([0.01,0.12,1.0,2.2,5.0])
        self.assertTrue(type(DUST.curve(wavelengths)),np.ndarray)
        [self.assertTrue(type(DUST.curve(w)),float) for w in wavelengths]
        return

if __name__ == "__main__":
    unittest.main()
