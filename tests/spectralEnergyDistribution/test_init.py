#! /usr/bin/env python

import sys,os
import unittest
from galacticus.spectralEnergyDistribution import parseDatasetName

class TestInit(unittest.TestCase):

    def test_ParseDatasetName(self):
        # Test bad names
        names = ["spectralEnergyDistribution:500.0_10000.0_200.0:rest:z1.000",
                 "totalSED:500.0_10000.0_200.0:rest:z1.000",
                 "totalSpectralEnergyDistribution:500.0_10000.0:observed:z1.000",
                 "totalSpectralEnergyDistribution:500.0_10000.0_200.0:obs:z1.000"]
        for name in names:
            MATCH = parseDatasetName(name)
            self.assertIsNone(MATCH)
        # Test good names
        # Test 1:
        name = "totalSpectralEnergyDistribution:500.0_10000.0_200.0:rest:z1.000:fixedWidth100.0"
        MATCH = parseDatasetName(name)
        self.assertIsNotNone(MATCH)
        self.assertEqual(MATCH.group("component"),"total")
        self.assertEqual(MATCH.group("wavelengths"),"500.0_10000.0_200.0")
        self.assertEqual(MATCH.group("lowerWavelength"),"500.0")
        self.assertEqual(MATCH.group("upperWavelength"),"10000.0")
        self.assertEqual(MATCH.group("resolution"),"200.0")
        self.assertEqual(MATCH.group("frame"),"rest")
        self.assertEqual(MATCH.group("redshift"),"1.000")
        self.assertEqual(MATCH.group("lineWidth"),":fixedWidth100.0")
        self.assertIsNone(MATCH.group("dust"))
        self.assertIsNone(MATCH.group("recent"))
        self.assertIsNone(MATCH.group("snrString"))
        self.assertIsNone(MATCH.group("snr"))        
        self.assertIsNone(MATCH.group("noLines"))        
        # Test 2:
        name = "totalSpectralEnergyDistribution:500.0_10000.0_200.0:rest:z1.000:snr10.0:recent"
        MATCH = parseDatasetName(name)
        self.assertIsNotNone(MATCH)
        self.assertEqual(MATCH.group("component"),"total")
        self.assertEqual(MATCH.group("wavelengths"),"500.0_10000.0_200.0")
        self.assertEqual(MATCH.group("lowerWavelength"),"500.0")
        self.assertEqual(MATCH.group("upperWavelength"),"10000.0")
        self.assertEqual(MATCH.group("resolution"),"200.0")
        self.assertEqual(MATCH.group("frame"),"rest")
        self.assertEqual(MATCH.group("redshift"),"1.000")
        self.assertEqual(MATCH.group("snrString"),":snr10.0")
        self.assertEqual(MATCH.group("snr"),"10.0")
        self.assertEqual(MATCH.group("recent"),":recent")
        self.assertIsNone(MATCH.group("dust"))
        self.assertIsNone(MATCH.group("lineWidth"))
        self.assertIsNone(MATCH.group("noLines"))        
        # Test 3
        name = "spheroidSpectralEnergyDistribution:255_10000.0_50:observed:z1.500:snr10:dustAtlas"
        MATCH = parseDatasetName(name)
        self.assertIsNotNone(MATCH)
        self.assertEqual(MATCH.group("component"),"spheroid")
        self.assertEqual(MATCH.group("wavelengths"),"255_10000.0_50")
        self.assertEqual(MATCH.group("lowerWavelength"),"255")
        self.assertEqual(MATCH.group("upperWavelength"),"10000.0")
        self.assertEqual(MATCH.group("resolution"),"50")
        self.assertEqual(MATCH.group("frame"),"observed")
        self.assertEqual(MATCH.group("redshift"),"1.500")
        self.assertEqual(MATCH.group("dust"),":dustAtlas")
        self.assertEqual(MATCH.group("snrString"),":snr10")
        self.assertEqual(MATCH.group("snr"),"10")
        self.assertIsNone(MATCH.group("recent"))
        self.assertIsNone(MATCH.group("noLines"))        
        # Test 4
        name = "spheroidSpectralEnergyDistribution:255_10000.0_50:observed:z1.500:dustAtlas:noLines"
        MATCH = parseDatasetName(name)
        self.assertIsNotNone(MATCH)
        self.assertEqual(MATCH.group("component"),"spheroid")
        self.assertEqual(MATCH.group("wavelengths"),"255_10000.0_50")
        self.assertEqual(MATCH.group("lowerWavelength"),"255")
        self.assertEqual(MATCH.group("upperWavelength"),"10000.0")
        self.assertEqual(MATCH.group("resolution"),"50")
        self.assertEqual(MATCH.group("frame"),"observed")
        self.assertEqual(MATCH.group("redshift"),"1.500")
        self.assertEqual(MATCH.group("dust"),":dustAtlas")
        self.assertEqual(MATCH.group("noLines"),":noLines")
        self.assertIsNone(MATCH.group("snrString"))
        self.assertIsNone(MATCH.group("snr"))        
        self.assertIsNone(MATCH.group("recent"))
        return
        
        

if __name__ == "__main__":
    unittest.main()


