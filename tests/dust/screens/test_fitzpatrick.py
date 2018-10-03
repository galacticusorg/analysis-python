#! /usr/bin/env python

import os,sys
import unittest
import numpy as np
from galacticus.dust.screens.manager import ScreenLaw
from galacticus.dust.screens.fitzpatrick import Fitzpatrick
from galacticus import rcParams

class TestFitzpatrick(unittest.TestCase):
    
    def test_Fitzpatrick(self):
        rcParams.update("dustFitzpatrick","Rv",2.71)
        DUST = Fitzpatrick()
        self.assertEqual(DUST.attrs["Rv"],2.71)
        self.assertIsNotNone(DUST.curve)
        wavelengths = np.array([0.01,0.12,1.0,10.0,20.0])
        self.assertTrue(type(DUST.curve(wavelengths)),np.ndarray)
        [self.assertTrue(type(DUST.curve(w)),float) for w in wavelengths]
        return

if __name__ == "__main__":
    unittest.main()
