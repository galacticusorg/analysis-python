#! /usr/bin/env python

import sys,os
import fnmatch
import numpy as np
import unittest
import warnings
from shutil import copyfile
from galacticus import rcParams
from galacticus.Cloudy import CloudyTable
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.constants import luminositySolar
from galacticus.constants import luminosityAB,erg
from galacticus.constants import mega,centi,parsec
from galacticus.constants import Pi
from galacticus.emissionLines.fluxes import EmissionLineFlux,ergPerSecondPerCentimeterSquared


class TestFluxes(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        DATA = GalacticusData()
        self.snapshotFile = DATA.searchDynamic("galacticus.snapshotExample.hdf5")
        self.lightconeFile = DATA.searchDynamic("galacticus.lightconeExample.hdf5")
        self.removeSnapshotExample = False
        self.removeLightconeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.snapshotFile is None:
            self.snapshotFile = DATA.dynamic+"/examples/galacticus.snapshotExample.hdf5"
            self.removeSnapshotExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.snapshotExample.hdf5",self.snapshotFile)
        if self.lightconeFile is None:
            self.lightconeFile = DATA.dynamic+"/examples/galacticus.lightconeExample.hdf5"
            self.removeLightconeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.lightconeExample.hdf5",self.lightconeFile)
        # Initialize the Totals class.
        GH5 = GalacticusHDF5(self.lightconeFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.LINES = EmissionLineFlux(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.LINES.galaxies.GH5Obj.close()
        del self.LINES
        if self.removeSnapshotExample:
            os.remove(self.snapshotFile)
        if self.removeLightconeExample:
            os.remove(self.lightconeFile)
        return

    def test_FluxesMatches(self):
        # Tests for correct dataset names
        for line in self.LINES.CLOUDY.listAvailableLines():
            for component in ["disk","spheroid"]:
                name = component+"LineFlux:"+line+":rest:z1.000"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineFlux:"+line+":observed:SDSS_r:z1.000"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineFlux:"+line+":observed:z1.000:recent"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineFlux:"+line+":rest:SDSS_g:z1.000:recent"
                self.assertTrue(self.LINES.matches(name))
        # Tests for incorrect dataset names
        name = "diskLineFlux:notAnEmissionLine:rest:z1.000"
        self.assertFalse(self.LINES.matches(name,raiseError=False))
        self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
        for name in ["totalLineFlux:balmerAlpha6563:rest:z1.000",
                     "diskLineFlux:SDSS_r:rest:z1.000",
                     "diskLineFlux:balmerAlpha6563:obs:z1.000",
                     "diskLineFlux:balmerAlpha6563:observed:1.000",
                     "diskLineFlux:balmerAlpha6563:rest:z1.000:dustAtlas",
                     "diskLineFlux:balmerAlpha6563:z1.000"]:
            self.assertFalse(self.LINES.matches(name,raiseError=False))
            self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
        return

    def test_FluxesGet(self):
        # Check bad names        
        redshift = 1.0
        name = "totalLineFlux:balmerAlpha6563:rest:z1.000"
        with self.assertRaises(RuntimeError):
            DATA = self.LINES.get(name,redshift)
        # Check values
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        component = "disk"
        for line in self.LINES.CLOUDY.listAvailableLines()[:1]:
            fluxName = component+"LineFlux:"+line+":rest:"+zStr                        
            luminosityName = component+"LineLuminosity:"+line+":rest:"+zStr                        
            GALS = self.LINES.galaxies.get(redshift,properties=["redshift",luminosityName])
            luminosityDistance = self.LINES.galaxies.GH5Obj.cosmology.luminosity_distance(GALS["redshift"].data)
            flux = GALS[luminosityName].data/(4.0*Pi*luminosityDistance**2)
            DATA = self.LINES.get(fluxName,redshift)
            self.assertEqual(DATA.name,fluxName)
            self.assertTrue(np.array_equal(flux,DATA.data))
        # Check error raised for snapshot output
        
        return

    def test_ergPerSecondPerCentimeterSquared(self):
        flux0 = np.random.rand(50)*0.04 + 0.01
        # Check conversion
        flux = np.log10(np.copy(flux0))
        flux += np.log10(luminositySolar)
        flux -= np.log10(erg)
        flux -= np.log10((mega*parsec/centi)**2)
        flux = 10.0**flux
        self.assertTrue(np.array_equal(flux,ergPerSecondPerCentimeterSquared(flux0)))
        return


if __name__ == "__main__":
    unittest.main()


