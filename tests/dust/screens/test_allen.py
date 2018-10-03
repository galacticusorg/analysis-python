#! /usr/bin/env python

import os,sys
import unittest
import numpy as np
from galacticus.dust.screens.manager import ScreenLaw
from galacticus.dust.screens.allen import Allen
from galacticus import rcParams

class TestAllen(unittest.TestCase):
    
    def test_Allen(self):
        rcParams.update("dustAllen","Rv",3.09)
        DUST = Allen()
        self.assertEqual(DUST.attrs["Rv"],3.09)
        self.assertIsNotNone(DUST.curve)
        wavelengths = np.array([0.01,0.1,3.0,10.0,20.0])
        self.assertTrue(type(DUST.curve(wavelengths)),np.ndarray)
        [self.assertTrue(type(DUST.curve(w)),float) for w in wavelengths]
        return

if __name__ == "__main__":
    unittest.main()
