#! /usr/bin/env python

import sys,os
import fnmatch
import numpy as np
import unittest
import six
if six.PY3:
    from unittest.mock import patch
else:
    from mock import patch
import warnings
from shutil import copyfile
from galacticus import rcParams
from galacticus.Cloudy import CloudyTable
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.datasets import Dataset
from galacticus.constants import angstrom
from galacticus.constants import mega,kilo
from galacticus.constants import Pi,speedOfLight
from galacticus.emissionLines.fullWidthHalfMaximum import FullWidthHalfMaximum


class TestFullWidthHalfMaximum(unittest.TestCase):
    
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
        self.FWHM = FullWidthHalfMaximum(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.FWHM.galaxies.GH5Obj.close()
        del self.FWHM
        if self.removeExample:
            os.remove(self.snapshotFile)
        return


    def test_FullWidthHalfMaximumGet(self):
        # Test crashes for incorrect dataset name
        with self.assertRaises(RuntimeError):
            name = "fullWidthHalfMaximum:notAnEmissionLine:dispersionWidth:z1.000"
            self.FWHM.matches(name,raiseError=True)
        # Test retrieval of dataset
        redshift = 1.0
        zStr = self.FWHM.galaxies.GH5Obj.getRedshiftString(redshift)
        N = self.FWHM.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        # i) using velocity dispersion
        name = "fullWidthHalfMaximum:balmerAlpha6563:dispersionWidth:"+zStr
        path = "galacticus.emissionLines.fullWidthHalfMaximum.FullWidthHalfMaximum.getVelocityWidth"
        with patch(path) as mocked_velocity:
            mocked_velocity.return_value = np.random.rand(N)*100.0
            restWavelength = self.FWHM.CLOUDY.getWavelength("balmerAlpha6563")
            widthVelocity = self.FWHM.getVelocityWidth(name,redshift)
            c = speedOfLight/kilo
            truth = restWavelength*(widthVelocity/c)
            DATA = self.FWHM.get(name,redshift)
            self.assertEqual(DATA.name,name)
            self.assertEqual(DATA.attr["unitsInSI"],angstrom)
            diff = np.fabs(DATA.data-truth)
            [self.assertLessEqual(d,1.0e-6) for d in diff]
        # ii) using fixed width
        name = "fullWidthHalfMaximum:balmerAlpha6563:fixedWidth102.03:"+zStr
        restWavelength = self.FWHM.CLOUDY.getWavelength("balmerAlpha6563")
        widthVelocity = np.ones(N)*102.03
        c = speedOfLight/kilo
        truth = restWavelength*(widthVelocity/c)
        DATA = self.FWHM.get(name,redshift)
        self.assertEqual(DATA.name,name)
        self.assertEqual(DATA.attr["unitsInSI"],angstrom)
        diff = np.fabs(DATA.data-truth)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        return


    def test_FullWidthHalfMaximumGetApproximateVelocityDispersion(self):
        # Test calculation of approximate velocity dispersion
        redshift = 1.0
        # Patch inclination to avoid generating random numbers
        with patch("galacticus.inclination.Inclination.get") as mocked_inc:
            inclination = 90.0
            DATA = Dataset(name="inclination",data=inclination,attr={"degrees":True})            
            mocked_inc.return_value = DATA
            # i) compute velocity dispersion
            scaleVelocityRatio = rcParams.getfloat("velocityDispersion","scaleVelocityRatio")
            minVelocityDipserion = rcParams.getfloat("velocityDispersion","minVelocityDipserion")
            properties = ["spheroidVelocity","diskVelocity","inclination"]
            GALS = self.FWHM.galaxies.get(redshift,properties)
            approximateVelocityDispersion = np.copy(GALS["spheroidVelocity"].data)
            baryonicSpheroidToTotalRatio = self.FWHM.getBaryonicBulgeToTotalRatio(redshift)
            emptyHalos = 999.9
            mask = baryonicSpheroidToTotalRatio == emptyHalos
            np.place(approximateVelocityDispersion,mask,minVelocityDipserion)
            diskDominated = baryonicSpheroidToTotalRatio<0.5
            if any(diskDominated):
                diskVelocity = np.copy(GALS["diskVelocity"].data)
                inclination = np.copy(GALS["inclination"].data)
                degrees = rcParams.getboolean("inclination","degrees")
                if degrees:
                    inclination *= (Pi/180.0)
                diskVelocity *= np.sqrt(np.sin(inclination)**2+(scaleVelocityRatio*np.cos(inclination))**2)
                np.place(approximateVelocityDispersion,diskDominated,diskVelocity[diskDominated])
            # ii) get value from function
            value = self.FWHM.getApproximateVelocityDispersion(redshift)
            # iii) check for any difference within some tolerance
            diff = np.fabs(approximateVelocityDispersion-value)
            [self.assertLessEqual(d,1.0e-6) for d in diff]
        return

    def test_FullWidthHalfMaximumGetBaryonicBulgeToTotalRatio(self):
        # Test calculation of baryon bulge-to-total ratio
        # i) compute bulge-to-total ratio
        redshift = 1.00
        emptyHalos = 999.9
        properties = ["diskMassStellar","spheroidMassStellar",
                      "diskMassGas","spheroidMassGas"]
        GALS = self.FWHM.galaxies.get(redshift,properties=properties)
        baryonicSpheroidMass = np.copy(GALS["spheroidMassStellar"].data+GALS["spheroidMassGas"].data)
        baryonicDiskMass = np.copy(GALS["diskMassStellar"].data+GALS["diskMassGas"].data)
        totalBaryonicMass = baryonicSpheroidMass + baryonicDiskMass
        mask = totalBaryonicMass == 0.0
        np.place(totalBaryonicMass,mask,1.0)
        np.place(baryonicSpheroidMass,mask,emptyHalos)
        trueRatio = baryonicSpheroidMass/totalBaryonicMass
        # ii) get value from function
        ratio = self.FWHM.getBaryonicBulgeToTotalRatio(redshift)
        # iii) check for any difference within some tolerance
        diff = np.fabs(trueRatio-ratio)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        return

    def test_FullWidthHalfMaximumGetVelocityWidth(self):
        redshift = 1.0
        zStr = self.FWHM.galaxies.GH5Obj.getRedshiftString(redshift)
        name = "fullWidthHalfMaximum:balmerAlpha6563:fixedWidth100.0:"+zStr
        N = self.FWHM.galaxies.GH5Obj.countGalaxiesAtRedshift(redshift)
        truth = np.ones(N)*100.0
        value = self.FWHM.getVelocityWidth(name,redshift)
        diff = np.fabs(truth-value)
        [self.assertLessEqual(d,1.0e-6) for d in diff]       
        name = "fullWidthHalfMaximum:balmerAlpha6563:dispersionWidth:"+zStr
        path = "galacticus.emissionLines.fullWidthHalfMaximum.FullWidthHalfMaximum.getApproximateVelocityDispersion"
        with patch(path) as mocked_velocity:
            truth = np.random.rand(N)*100.0
            mocked_velocity.return_value = truth
            value = self.FWHM.getVelocityWidth(name,redshift)
            diff = np.fabs(truth-value)
            [self.assertLessEqual(d,1.0e-6) for d in diff]       
        return





    def test_FullWidthHalfMaximumMatches(self):
        # Tests for correct dataset names
        for line in self.FWHM.CLOUDY.listAvailableLines():
            name = "fullWidthHalfMaximum:"+line+":dispersionWidth:z1.000"
            self.assertTrue(self.FWHM.matches(name))
            name = "fullWidthHalfMaximum:"+line+":dispersionWidth:z1.000:recent"
            self.assertTrue(self.FWHM.matches(name))
            name = "fullWidthHalfMaximum:"+line+":fixedWidth100.0:z1.000"
            self.assertTrue(self.FWHM.matches(name))
            name = "fullWidthHalfMaximum:"+line+":fixedWidth10:z1.000"
            self.assertTrue(self.FWHM.matches(name))
        name = "fullWidthHalfMaximum:balmerAlpha6563:dispersionWidth:z1.000"
        names = ["fullWidthHalfMaximum:notAnEmissionLine:dispersionWidth:z1.000",
                 "fullWidthHalfMaximum:balmerAlpha6563:dispersionWidth",
                 "fullWidthHalfMaximum:balmerAlpha6563:fixedWidth:z1.000",
                 "fullWidthHalfMaximum:balmerAlpha6563:fixed100.0:z1.000",
                 "fullWidthHalfMaximum:balmerAlpha6563:fixedWidth1.0e3:z1.000"]
        for name in names:
            self.assertFalse(self.FWHM.matches(name,raiseError=False))
            with self.assertRaises(RuntimeError):
                self.FWHM.matches(name,raiseError=True)
        return

    def test_FullWidthHalfMaximumParseDatasetName(self):
        name = "fullWidthHalfMaximum:balmerAlpha6563:dispersionWidth:z1.000"        
        MATCH = self.FWHM.parseDatasetName(name)
        self.assertIsNotNone(MATCH)
        self.assertEqual(MATCH.group('lineName'),'balmerAlpha6563')
        self.assertEqual(MATCH.group('width'),'dispersionWidth')
        self.assertEqual(MATCH.group('redshift'),'1.000')
        self.assertIsNone(MATCH.group('recent'))
        name = "fullWidthHalfMaximum:balmerAlpha6563:fixedWidth102.0:z1.010:recent"        
        MATCH = self.FWHM.parseDatasetName(name)
        self.assertIsNotNone(MATCH)
        self.assertEqual(MATCH.group('lineName'),'balmerAlpha6563')
        self.assertEqual(MATCH.group('width'),'fixedWidth102.0')
        self.assertEqual(MATCH.group('redshift'),'1.010')
        self.assertEqual(MATCH.group('recent'),':recent')
        names = ["fullWidthHalfMaximum:notAnEmissionLine:dispersionWidth:z1.000",
                 "fullWidthHalfMaximum:balmerAlpha6563:dispersionWidth",
                 "fullWidthHalfMaximum:balmerAlpha6563:fixedWidth:z1.000",
                 "fullWidthHalfMaximum:balmerAlpha6563:fixed100.0:z1.000",
                 "fullWidthHalfMaximum:balmerAlpha6563:fixedWidth1.0e3:z1.000"]
        for name in names:
            MATCH = self.FWHM.parseDatasetName(name)
            self.assertIsNone(MATCH)
        return



            
if __name__ == "__main__":
    unittest.main()


