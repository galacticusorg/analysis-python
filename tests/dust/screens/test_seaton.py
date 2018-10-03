#! /usr/bin/env python

import os,sys
import unittest
import numpy as np
from galacticus.dust.screens.manager import ScreenLaw
from galacticus.dust.screens.seaton import Seaton
from galacticus import rcParams

class TestSeaton(unittest.TestCase):
    
    def test_Seaton(self):
        rcParams.update("dustSeaton","Rv",3.09)
        DUST = Seaton()
        self.assertEqual(DUST.attrs["Rv"],3.09)
        self.assertIsNotNone(DUST.curve)
        wavelengths = np.array([0.01,0.12,1.0,10.0,20.0])
        self.assertTrue(type(DUST.curve(wavelengths)),np.ndarray)
        [self.assertTrue(type(DUST.curve(w)),float) for w in wavelengths]
        return

if __name__ == "__main__":
    unittest.main()
