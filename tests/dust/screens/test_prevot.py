#! /usr/bin/env python

import os,sys
import unittest
import numpy as np
from galacticus.dust.screens.manager import ScreenLaw
from galacticus.dust.screens.prevot import Prevot
from galacticus import rcParams

class TestPrevot(unittest.TestCase):
    
    def test_Prevot(self):
        rcParams.update("dustPrevot","Rv",3.09)
        DUST = Prevot()
        self.assertEqual(DUST.attrs["Rv"],3.09)
        self.assertIsNotNone(DUST.curve)
        wavelengths = np.array([0.01,0.1275,1.0,2.2,5.0])
        self.assertTrue(type(DUST.curve(wavelengths)),np.ndarray)
        [self.assertTrue(type(DUST.curve(w)),float) for w in wavelengths]
        return

if __name__ == "__main__":
    unittest.main()
