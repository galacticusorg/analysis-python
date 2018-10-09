#! /usr/bin/env python

import sys
import fnmatch
import numpy as np
import unittest
import warnings
from galacticus import rcParams
from galacticus.Cloudy import CloudyTable


class TestCloudyTable(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.CLOUDY = CloudyTable()
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory
        self.CLOUDY.close()
        del self.CLOUDY
        rcParams.reset()
        return

    def test_CloudyTableListAvailableLines(self):
        allLines = self.CLOUDY.lsDatasets("/lines")
        found = self.CLOUDY.listAvailableLines()
        self.assertEqual(allLines,found)
        return

    def test_CloudyTableLoadEmissionLine(self):
        name = "balmerAlpha6563"
        self.CLOUDY.loadEmissionLine(name)
        self.assertTrue(name in self.CLOUDY.lines.keys())
        LINE = self.CLOUDY.lines[name]
        self.assertEqual(LINE.name,name)
        wavelength = self.CLOUDY.readAttributes("lines/"+name,required=["wavelength"])["wavelength"]
        self.assertEqual(wavelength,LINE.wavelength)
        luminosities = self.CLOUDY.readDataset('/lines/'+name)
        self.assertEqual(luminosities.shape,LINE.luminosities.shape)
        diff = np.fabs(luminosities-LINE.luminosities).flatten()
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            badName = "notAnEmissionLine9999"
            self.CLOUDY.loadEmissionLine(badName)
            self.assertFalse(badName in self.CLOUDY.lines.keys())
        return

    def test_CloudyTableLoadEmissionLines(self):
        self.CLOUDY.lines = {}
        self.CLOUDY.loadEmissionLines()
        lines = self.CLOUDY.lsDatasets("/lines")
        self.assertEqual(len(lines),len(self.CLOUDY.lines.keys()))
        [self.assertTrue(line in self.CLOUDY.lines.keys()) for line in lines]
        [self.assertIsNotNone(self.CLOUDY.lines[l]) for l in lines]
        return

    def test_CloudyTableGetWavelength(self):
        self.CLOUDY.loadEmissionLines()
        for name in self.CLOUDY.lsDatasets("/lines"):
            value = self.CLOUDY.lines[name].wavelength
            wavelength = self.CLOUDY.getWavelength(name)
            self.assertEqual(value,wavelength)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            badName = "notAnEmissionLine9999"
            self.assertRaises(IndexError,self.CLOUDY.getWavelength,badName)
        return

    def test_CLoudyTableGetInterpolant(self):
        for name in self.CLOUDY.interpolants:
            data = self.CLOUDY.getInterpolant(name)
            values = np.log10(self.CLOUDY.readDataset('/'+name))
            diff = data - values
            [self.assertLessEqual(d,1.0e-6) for d in diff]
        with self.assertRaises(KeyError):
            self.CLOUDY.getInterpolant("someGas")        
        return

    def test_CloudyTableLoadInterpolantsData(self):
        self.CLOUDY.interpolantsData = None
        self.CLOUDY.loadInterpolantsData()
        self.assertIsNotNone(self.CLOUDY.interpolantsData)
        return

    def test_CloudyTablePrepareGalaxyData(self):
        N = 1000
        values = np.random.rand(N)
        data = self.CLOUDY.prepareGalaxyData(values,values,values,values,values)
        self.assertEqual(len(data),N)
        self.assertEqual(len(data[0]),5)
        with self.assertRaises(TypeError):
            self.CLOUDY.prepareGalaxyData(values)        
        return

    def test_CloudyTableInterpolate(self):
        # Test incorrect calling of function
        badName= "notAnEmissionLine999"
        N = 1000
        metallicity = np.random.rand(N)
        densityHydrogen = np.random.rand(N)
        ionizingFluxHydrogen = np.random.rand(N)
        ionizingFluxHeliumToHydrogen = np.random.rand(N)
        ionizingFluxOxygenToHydrogen = np.random.rand(N)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")        
            with self.assertRaises(KeyError):
                self.CLOUDY.interpolate(badName,metallicity,
                                        densityHydrogen,
                                        ionizingFluxHydrogen,
                                        ionizingFluxHeliumToHydrogen,
                                        ionizingFluxOxygenToHydrogen)
        with self.assertRaises(TypeError):
            name = "balmerAlpha6563"
            self.CLOUDY.interpolate(name,metallicity)
        # Generate random data to test correct calling
        i = self.CLOUDY.getInterpolant("metallicity")
        diff = i.max() - i.min()
        metallicity = np.random.rand(N)*diff + i.min()
        i = self.CLOUDY.getInterpolant("densityHydrogen")
        diff = i.max() - i.min()
        densityHydrogen = np.random.rand(N)*diff + i.min()
        i = self.CLOUDY.getInterpolant("ionizingFluxHydrogen")
        diff = i.max() - i.min()
        ionizingFluxHydrogen = np.random.rand(N)*diff + i.min()
        i = self.CLOUDY.getInterpolant("ionizingFluxHeliumToHydrogen")
        diff = i.max() - i.min()
        ionizingFluxHeliumToHydrogen = np.random.rand(N)*diff + i.min()
        i = self.CLOUDY.getInterpolant("ionizingFluxOxygenToHydrogen")
        diff = i.max() - i.min()
        ionizingFluxOxygenToHydrogen = np.random.rand(N)*diff + i.min()
        # Test interpolation
        luminosity = self.CLOUDY.interpolate(name,metallicity,
                                             densityHydrogen,
                                             ionizingFluxHydrogen,
                                             ionizingFluxHeliumToHydrogen,
                                             ionizingFluxOxygenToHydrogen)
        self.assertIsInstance(luminosity,np.ndarray)
        metallicity[0] *= 1000.0
        rcParams.update("cloudy","fill_value","nan")
        luminosity = self.CLOUDY.interpolate(name,metallicity,
                                             densityHydrogen,
                                             ionizingFluxHydrogen,
                                             ionizingFluxHeliumToHydrogen,
                                             ionizingFluxOxygenToHydrogen)
        self.assertTrue(np.any(np.isnan(luminosity)))
        rcParams.update("cloudy","bounds_error",True)
        self.assertRaises(ValueError,self.CLOUDY.interpolate,name,metallicity,
                          densityHydrogen,ionizingFluxHydrogen,
                          ionizingFluxHeliumToHydrogen,
                          ionizingFluxOxygenToHydrogen)
        return
        

if __name__ == "__main__":
    unittest.main()
