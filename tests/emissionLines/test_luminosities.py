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
from galacticus.constants import massSolar,luminositySolar,metallicitySolar
from galacticus.constants import luminosityAB,erg
from galacticus.constants import parsec,angstrom
from galacticus.constants import mega,centi
from galacticus.constants import Pi,speedOfLight
from galacticus.constants import massAtomic,atomicMassHydrogen,massFractionHydrogen
from galacticus.emissionLines.luminosities import EmissionLineLuminosity,ergPerSecond


class TestLuminosities(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # Locate the dynamic version of the galacticus.snapshotExample.hdf5 file.
        DATA = GalacticusData()
        self.snapshotFile = DATA.searchDynamic("galacticus.snapshotExample.hdf5")
        self.removeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.snapshotFile is None:
            self.snapshotFile = DATA.dynamic+"/examples/galacticus.snapshotExample.hdf5"
            self.removeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.snapshotExample.hdf5",self.snapshotFile)
        # Initialize the Totals class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.LINES = EmissionLineLuminosity(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.LINES.galaxies.GH5Obj.close()
        del self.LINES
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_LuminositiesGet(self):
        # Test return of emission line luminosities
        redshift = 1.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        for line in self.LINES.CLOUDY.listAvailableLines():
            name = "diskLineLuminosity:"+line+":rest:"+zStr
            DATA = self.LINES.get(name,redshift)
            self.assertIsNotNone(DATA)
            self.assertEqual(DATA.name,name)
            self.assertIsNotNone(DATA.data)
            self.assertIsInstance(DATA.data,np.ndarray)
        return

    def test_LuminositiesGetContinuumLuminosities(self):
        # Test that continuum luminosities returned
        redshift = 1.0
        OUT = self.LINES.galaxies.GH5Obj.selectOutput(redshift)
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        names = ["spheroidLineLuminosity:balmerAlpha6563:rest:"+zStr,\
                     "diskLineLuminosity:balmerBeta4861:observed:"+zStr]
        for name in names:
            Ly,He,O = self.LINES.getContinuumLuminosityNames(name)
            LUMS = self.LINES.getContinuumLuminosities(name,redshift)
            for name in [Ly,He,O]:
                self.assertIsNotNone(LUMS[name])
        return

    def test_LuminositiesGetContinuumLuminosityNames(self):
        # Tests that continuum luminosities names returned for luminosity name
        name = "diskLineLuminosity:balmerAlpha6563:rest:z1.000"
        Ly,He,O = self.LINES.getContinuumLuminosityNames(name)
        self.assertEqual(Ly,"diskLymanContinuumLuminosity:rest:z1.000")
        self.assertEqual(He,"diskHeliumContinuumLuminosity:rest:z1.000")
        self.assertEqual(O,"diskOxygenContinuumLuminosity:rest:z1.000")
        name = "spheroidLineLuminosity:balmerAlpha6563:observed:z1.000:recent"
        Ly,He,O = self.LINES.getContinuumLuminosityNames(name)
        self.assertEqual(Ly,"spheroidLymanContinuumLuminosity:observed:z1.000:recent")
        self.assertEqual(He,"spheroidHeliumContinuumLuminosity:observed:z1.000:recent")
        self.assertEqual(O,"spheroidOxygenContinuumLuminosity:observed:z1.000:recent")
        # Check Runtime error is returned for incorrect dataset anme
        name = "spheroidLineLuminosity:balmerAlpha6563:observed"
        self.assertRaises(RuntimeError,self.LINES.getContinuumLuminosityNames,name)
        name = "totalLineLuminosity:balmerAlpha6563:observed:z1.000"
        self.assertRaises(RuntimeError,self.LINES.getContinuumLuminosityNames,name)
        name = "diskLineLuminosity:balmerAlpha6563"
        self.assertRaises(RuntimeError,self.LINES.getContinuumLuminosityNames,name)
        return

    def test_LuminositiesGetIonizingFluxHydrogen(self):
        # Test for computing Hydrogen ionizing flux
        luminosity = np.random.rand(50)*1.0e20 + 1.0e3
        ionizingFluxHydrogen = np.copy(luminosity)
        np.place(ionizingFluxHydrogen,ionizingFluxHydrogen==0.0,np.nan)
        ionizingFluxHydrogen = np.log10(ionizingFluxHydrogen)+50.0
        result = self.LINES.getIonizingFluxHydrogen(luminosity)
        diff = np.fabs(result-ionizingFluxHydrogen)
        [self.assertLessEqual(d,1.0e06) for d in diff]
        return

    def test_LuminositiesGetIonizingFluxRatio(self):
        # Test for computing ionizing flux ratios
        lyman = np.random.rand(50)*1.0e20 + 1.0e3
        helium = np.random.rand(50)*1.0e20 + 1.0e3
        ratio = np.log10(helium/lyman)
        result = self.LINES.getIonizingFluxRatio(lyman,helium)
        diff = np.fabs(ratio-result)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        np.place(lyman,np.random.rand(50)<0.1,0.0)
        np.place(helium,np.random.rand(50)<0.1,0.0)
        mask = np.logical_or(lyman==0.0,helium==0.0)
        result = self.LINES.getIonizingFluxRatio(lyman,helium)
        for i in range(len(result)):
            self.assertEqual(mask[i],np.isnan(result[i]))
        return

    def test_LuminositiesGetLifetimeHIIRegions(self):
        # Test calculation of HII lifetimes
        lifetime = np.random.rand(1)[0]*1.0e-3
        rcParams.update("emissionLine","lifetimeHIIRegion",lifetime)
        diff = np.fabs(lifetime-self.LINES.getLifetimeHIIRegions())
        self.assertLessEqual(diff,1.0e-6)
        return

    def test_LuminositiesGetLuminosityMultiplier(self):
        # Test on calculation of luminosity multplier
        redshift = 1.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        for filterName in ["SDSS_r","SDSS_g"]:
            for frame in ["rest","observed"]:
                for line in ["balmerAlpha6563","balmerBeta4861"]:
                    name = "diskLineLuminosity:"+line+":"+frame+":"+filterName+":"+zStr
                    FILTER = self.LINES.GALFIL.load(filterName)
                    GALS = self.LINES.galaxies.get(redshift,properties=["redshift"])
                    lineWavelength = self.LINES.CLOUDY.getWavelength(line)
                    if frame == "observed":
                        lineWavelength *= (1.0+GALS['redshift'].data)
                    else:
                        lineWavelength *= np.ones_like(GALS['redshift'].data)
                    multiplier = FILTER.interpolate(lineWavelength)
                    multiplier /= FILTER.integrate()
                    if frame == "observed":
                        multiplier /= (1.0+GALS["redshift"].data)
                    else:
                        lineWavelength *= np.ones_like(GALS['redshift'].data)
                    result = self.LINES.getLuminosityMultiplier(name,redshift)
                    diff = np.fabs(multiplier-result)
                    [self.assertLessEqual(d,1.0e-6) for d in diff]
        # Test multiplier equals unity if no filter output
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr
        result = self.LINES.getLuminosityMultiplier(name,redshift)
        self.assertEqual(result,1.0)
        redshift = 2.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        name = "spheroidLineLuminosity:balmerAlpha6563:observed:"+zStr
        result = self.LINES.getLuminosityMultiplier(name,redshift)
        self.assertEqual(result,1.0)
        name = "diskLineLuminosity:balmerAlpha6563:SDSS_r:"+zStr
        self.assertRaises(RuntimeError,self.LINES.getLuminosityMultiplier,\
                              name,redshift)
        name = "diskLineLuminosity:balmerAlpha6563:rest:SDSS_h:"+zStr
        self.assertRaises(RuntimeError,self.LINES.getLuminosityMultiplier,\
                              name,redshift)
        return


    def test_LuminositiesGetMassHIIRegions(self):
        # Test for computation for mass of HII regions
        mass = np.random.rand(1)[0]*1.0e3
        rcParams.update("emissionLine","massHIIRegion",mass)
        diff = np.fabs(mass-self.LINES.getMassHIIRegions())
        self.assertLessEqual(diff,1.0e-6)
        return

    def test_LuminositiesGetNumberHIIRegions(self):
        # Test computation of number of HII regions
        redshift = 1.0
        OUT = self.LINES.galaxies.GH5Obj.selectOutput(redshift)
        for component in ["disk","spheroid"]:
            sfr = np.array(OUT["nodeData/"+component+"StarFormationRate"])
            n = sfr*self.LINES.getLifetimeHIIRegions()
            n /= self.LINES.getMassHIIRegions()
            N = self.LINES.getNumberHIIRegions(redshift,component)
            diff = np.fabs(n-N)
            [self.assertLessEqual(d,1.0e-6) for d in diff]
        self.assertRaises(ValueError,self.LINES.getNumberHIIRegions,redshift,"total")
        return

    def test_LuminositiesLineInCloudyOutput(self):
        # Test line name is found in Cloudy library
        for line in self.LINES.CLOUDY.listAvailableLines():
            self.assertTrue(self.LINES.lineInCloudyOutput(line))
        self.assertFalse(self.LINES.lineInCloudyOutput("notAnEmissionLine"))
        return

    def test_LuminositiesMatches(self):
        # Tests for correct dataset names
        for line in self.LINES.CLOUDY.listAvailableLines():
            for component in ["disk","spheroid"]:
                name = component+"LineLuminosity:"+line+":rest:z1.000"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineLuminosity:"+line+":observed:SDSS_r:z1.000"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineLuminosity:"+line+":observed:z1.000:recent"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineLuminosity:"+line+":rest:SDSS_g:z1.000:recent"
                self.assertTrue(self.LINES.matches(name))
        # Tests for incorrect dataset names
        name = "diskLineLuminosity:notAnEmissionLine:rest:z1.000"
        self.assertFalse(self.LINES.matches(name,raiseError=False))
        self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
        for name in ["totalLineLuminosity:balmerAlpha6563:rest:z1.000",
                     "diskLineLuminosity:SDSS_r:rest:z1.000",
                     "diskLineLuminosity:balmerAlpha6563:obs:z1.000",
                     "diskLineLuminosity:balmerAlpha6563:observed:1.000",
                     "diskLineLuminosity:balmerAlpha6563:rest:z1.000:dustAtlas",
                     "diskLineLuminosity:balmerAlpha6563:z1.000"]:
            self.assertFalse(self.LINES.matches(name,raiseError=False))
            self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
        return

    def test_LuminositiesGet(self):
        # Check bad names        
        redshift = 1.0
        name = "totalLineLuminosity:balmerAlpha6563:rest:z1.000"
        with self.assertRaises(RuntimeError):
            DATA = self.LINES.get(name,redshift)
        # Check values
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        component = "disk"
        for line in self.LINES.CLOUDY.listAvailableLines()[:1]:
            name = component+"LineLuminosity:"+line+":rest:"+zStr                        
            LymanName,HeliumName,OxygenName = self.LINES.getContinuumLuminosityNames(name)
            FLUXES = self.LINES.getContinuumLuminosities(name,redshift)
            if any([FLUXES[name] is None for name in FLUXES.keys()]):
                warnings.warn(funcname+"(): Unable to compute emission line luminosity as one of the "+\
                                  "continuum luminosities is missing. Returning None instance.")
                luminosity = None
            else:
                metals = component+"GasMetallicity"
                hydrogen = component+"HydrogenGasDensity"
                GALS = self.LINES.galaxies.get(redshift,properties=[metals,hydrogen])
                hydrogenGasDensity = np.log10(np.copy(GALS[hydrogen].data))
                metallicity = np.log10(np.copy(GALS[metals].data)+1.0e-50)
                del GALS
                ionizingFluxHydrogen = self.LINES.getIonizingFluxHydrogen(FLUXES[LymanName].data)
                numberHIIRegion = self.LINES.getNumberHIIRegions(redshift,component)
                mask = numberHIIRegion == 0.0
                numberHIIRegion[mask] = 1.0
                ionizingFluxHydrogen -= np.log10(numberHIIRegion)
                ionizingFluxHeliumToHydrogen = self.LINES.getIonizingFluxRatio(FLUXES[LymanName].data,FLUXES[HeliumName].data)
                ionizingFluxOxygenToHelium = self.LINES.getIonizingFluxRatio(FLUXES[HeliumName].data,FLUXES[OxygenName].data)
                luminosity = np.copy(self.LINES.CLOUDY.interpolate(line,metallicity,hydrogenGasDensity,
                                                                   ionizingFluxHydrogen,ionizingFluxHeliumToHydrogen,
                                                                   ionizingFluxOxygenToHelium))
                luminosityMultiplier = self.LINES.getLuminosityMultiplier(name,redshift)
                luminosity *= (luminosityMultiplier*numberHIIRegion*erg/luminositySolar)
                zeroCorrection = rcParams.getfloat("emissionLine","zeroCorrection",fallback=1.0e-50)
                luminosity += zeroCorrection
            DATA = self.LINES.get(name,redshift)
            if luminosity is None:
                self.assertIsNone(DATA.data)
            else:
                diff = np.fabs(DATA.data-luminosity)
                [self.assertLessEqual(d,1.0e-6) for d in diff if np.invert(np.isnan(d))]
                maskD = np.isnan(DATA.data)
                mask0 = np.isnan(luminosity)
                [self.assertEqual(m,n) for m,n in zip(maskD,mask0)]
        return

    def test_ergPerSecond(self):
        luminosity0 = np.random.rand(50)*4.0 + 3.0
        luminosity0 = 10.0**luminosity0
        # Check conversion        
        luminosity = np.copy(luminosity0)
        luminosity *= luminositySolar/erg
        self.assertTrue(np.array_equal(luminosity,ergPerSecond(luminosity0)))
        return


if __name__ == "__main__":
    unittest.main()


