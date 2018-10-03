#! /usr/bin/env python

import re
import os,sys
import unittest
import numpy as np
from shutil import copyfile
from galacticus.galaxies import Galaxies
from galacticus.io import GalacticusHDF5
from galacticus.data import GalacticusData
from galacticus.dust.dustScreens import DustScreen


class TestDustScreen(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Locate the dynamic version of the galacticus.snapshotExample.hdf5 file.
        DATA = GalacticusData(verbose=False)
        self.snapshotFile = DATA.searchDynamic("galacticus.snapshotExample.hdf5")
        self.removeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.snapshotFile is None:
            self.snapshotFile = DATA.dynamic+"/examples/galacticus.snapshotExample.hdf5"
            self.removeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.snapshotExample.hdf5",self.snapshotFile)
        # Initialize the DustScreen class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.DUST = DustScreen(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.DUST.galaxies.GH5Obj.close()
        del self.DUST
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def test_DustScreenListAvailableScreens(self):
        self.assertEqual(self.DUST.listAvailableScreens(),
                         self.DUST.SCREENS.laws.keys())
        return
    
    def test_DustScreenParseDatasetName(self):
        for screen in self.DUST.listAvailableScreens():
            for component in ["disk","spheroid"]:
                for frame in ["rest","observed"]:
                    name = component+"LuminositiesStellar:SDSS_r:"+frame+":z1.000:dust"+screen
                    MATCH = self.DUST.parseDatasetName(name)
                    self.assertIsNotNone(MATCH)
                    self.assertEqual(MATCH.group("screen"),screen)
                    self.assertIsNone(MATCH.group("av"))                    
                    name = component+"LineLuminosity:balmerAlpha6563:"+frame+":SDSS_r:z1.000:dust"+screen
                    MATCH = self.DUST.parseDatasetName(name)
                    self.assertIsNotNone(MATCH)
                    self.assertEqual(MATCH.group("screen"),screen)
                    self.assertEqual(MATCH.group("filterName"),":SDSS_r")
                    self.assertIsNone(MATCH.group("av"))                    
                    name = component+"LineLuminosity:balmerAlpha6563:"+frame+":SDSS_r:z1.000:dust"+screen+"_Av0.1"
                    MATCH = self.DUST.parseDatasetName(name)
                    self.assertIsNotNone(MATCH)
                    self.assertEqual(MATCH.group("screen"),screen)
                    self.assertEqual(MATCH.group("filterName"),":SDSS_r")
                    self.assertEqual(MATCH.group("av"),"0.1")
                    name = component+"LineLuminosity:balmerAlpha6563:"+frame+":z1.000:dust"+screen
                    MATCH = self.DUST.parseDatasetName(name)
                    self.assertIsNotNone(MATCH)
                    self.assertEqual(MATCH.group("screen"),screen)
                    self.assertIsNone(MATCH.group("filterName"))
            name = "totalLineLuminositiy:balmerAlpha6563:rest:z1.000:"+screen
            MATCH = self.DUST.parseDatasetName(name)
            self.assertIsNone(MATCH)
            name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustClazeti"
            MATCH = self.DUST.parseDatasetName(name)
            self.assertIsNone(MATCH)
            name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCompendium"
            MATCH = self.DUST.parseDatasetName(name)
            self.assertIsNone(MATCH)
            name = "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti_Av"
            MATCH = self.DUST.parseDatasetName(name)
            self.assertIsNone(MATCH)
        return
    
    def test_DustScreenMatches(self):
        for screen in self.DUST.listAvailableScreens():
            for component in ["disk","spheroid"]:
                for frame in ["rest","observed"]:
                    name = component+"LuminositiesStellar:SDSS_r:"+frame+":z1.000:dust"+screen
                    self.assertTrue(self.DUST.matches(name))
                    name = component+"LineLuminosity:balmerAlpha6563:"+frame+":SDSS_r:z1.000:dust"+screen+"_Av0.1"
                    self.assertTrue(self.DUST.matches(name))
                    name = component+"LineLuminosity:balmerAlpha6563:"+frame+":z1.000:dust"+screen
                    self.assertTrue(self.DUST.matches(name))
            name = "totalLineLuminositiy:balmerAlpha6563:rest:z1.000:dust"+screen
            self.assertFalse(self.DUST.matches(name,raiseError=False))
            with self.assertRaises(RuntimeError):
                self.DUST.matches(name,raiseError=True)
        for name in ["diskLuminositiesStellar:SDSS_g:rest:z1.000:dustClazeti",
                     "diskLuminositiesStellar:SDSS_g:rest:z1.000:dustCompendium",
                     "diskLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti_Av"]:
            self.assertFalse(self.DUST.matches(name,raiseError=False))
            with self.assertRaises(RuntimeError):
                self.DUST.matches(name,raiseError=True)
        return

    def test_DustScreenGetDustFreeName(self):
        name = "diskLuminositiesStellar:SDSS_r:rest:z1.000"
        self.assertEqual(self.DUST.getDustFreeName(name+":dustCalzetti"),name)
        self.assertEqual(self.DUST.getDustFreeName(name+":dustCalzetti_Av0.1"),name)
        with self.assertRaises(RuntimeError):
            self.DUST.getDustFreeName(name+":Calzetti_Av")
            self.DUST.getDustFreeName(name+":Clazeti")
        return

    def test_DustScreenGetDustFreeLuminosity(self):
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        OUT = self.DUST.galaxies.GH5Obj.selectOutput(redshift)
        for name in ["diskLuminositiesStellar:SDSS_r:rest:"+zStr,
                     "diskLineLuminosity:balmerAlpha6563:rest:"+zStr]:
            DATA = self.DUST.getDustFreeLuminosity(name+":dustCalzetti",redshift)
            self.assertIsNotNone(DATA)
            self.assertEqual(DATA.name,name)
            self.assertIsNotNone(DATA.data)
        for name in ["diskLuminositiesStellar:SDSS_r:rest:"+zStr+":Calzetti",
                     "diskLineLuminosity:balmerAlpha6563:rest:"+zStr+":Clazeti"]:      
            with self.assertRaises(RuntimeError):
                self.DUST.getDustFreeLuminosity(name,redshift)
        return

    def test_DustScreenSelectDustScreen(self):
        for screen in self.DUST.listAvailableScreens():
            OBJ = self.DUST.selectDustScreen(screen)
            self.assertEqual(OBJ.__class__.__name__,screen)
        with self.assertRaises(KeyError):
            self.DUST.selectDustScreen("Clazeti")
            self.DUST.selectDustScreen("Compendium")
        return

    
    def test_DustScreenGetAv(self):
        for name in ["diskLuminositiesStellar:SDSS_g:rest:z1.000:dustClazeti",
                     "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti",
                     "diskLuminositiesStellar:SDSS_g:rest:z1.000:dustCompendium",
                     "diskLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti_Av"]:
            with self.assertRaises(RuntimeError):
                self.DUST.get(name,1.0)
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        for screen in self.DUST.listAvailableScreens():
            for component in ["disk","spheroid"]:
                for frame in ["rest","observed"]:
                    name = component+"LuminositiesStellar:SDSS_r:"+frame+":"+zStr+":dust"+screen
                    self.assertIsNotNone(self.DUST.getAv(name,redshift))
                    name = component+"LuminositiesStellar:SDSS_r:"+frame+":"+zStr+":dust"+screen+"_Av0.1"
                    self.assertTrue(all(self.DUST.getAv(name,redshift)==0.1))
        return

    def test_DustScrrensGet(self):
        for name in ["diskLuminositiesStellar:SDSS_g:rest:z1.000:dustClazeti",
                     "totalLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti",
                     "diskLuminositiesStellar:SDSS_g:rest:z1.000:dustCompendium",
                     "diskLuminositiesStellar:SDSS_g:rest:z1.000:dustCalzetti_Av"]:
            with self.assertRaises(RuntimeError):
                DATA = self.DUST.get(name,1.0)
        redshift = 1.0
        zStr = self.DUST.galaxies.GH5Obj.getRedshiftString(redshift)
        name = "diskLuminositiesStellar:SDSS_r:rest:"+zStr+":dustCalzetti"
        DATA = self.DUST.get(name,redshift)
        self.assertEqual(DATA.name,name)
        self.assertIsNotNone(DATA.data)
        name = "diskLuminositiesStellar:SDSS_r:observed:"+zStr+":dustAllen_Av0.4"
        DATA = self.DUST.get(name,redshift)
        self.assertEqual(DATA.name,name)
        self.assertIsNotNone(DATA.data)
        return

if __name__ == "__main__":
    unittest.main()
