#! /usr/bin/env python

import unittest
from unittest.mock import patch
import sys,os,fnmatch
import warnings
from shutil import copyfile
from galacticus import rcParams
from galacticus.data import locateDatasetsRepository
from galacticus.data import recursiveGlob
from galacticus.data import GalacticusData


class TestLocateDatasetsRepository(unittest.TestCase):
    
    def test_LocateDatasetsRepository(self):        
        with patch("galacticus.rcParams.get") as mocked_rc:
            mocked_rc.return_value = None
            with self.assertRaises(RuntimeError):
                locateDatasetsRepository()
            mocked_rc.return_value = "/home/Galacticus/datasets"
            path = locateDatasetsRepository()
            self.assertIsNotNone(path)
            self.assertEqual(path,"/home/Galacticus/datasets/")
        return

    
class TestRecursiveGlob(unittest.TestCase):
    
    def test_RecursiveGlob(self):        
        path = locateDatasetsRepository()
        result = recursiveGlob(path,"SDSS_r.xml")
        self.assertEqual(len(result),1)
        self.assertTrue(result[0].endswith("datasets/static/filters/SDSS_r.xml"))
        result = recursiveGlob(path,"SDSS_A.xml")
        self.assertEqual(len(result),0)
        return
                        
    
class TestGalacticusData(unittest.TestCase):
    
    def test_GalacticusDataInit(self):
        with patch("galacticus.data.locateDatasetsRepository") as mocked_locate:
            mocked_locate.return_value = "None"
            with self.assertRaises(RuntimeError):
                DATA = GalacticusData()
            mocked_locate.return_value = "."
            with self.assertRaises(RuntimeError):
                DATA = GalacticusData()
        DATA = GalacticusData()
        self.assertIsInstance(DATA,GalacticusData)
        return

    def test_GalacticusData_searchDirectory(self):
        DATA = GalacticusData()
        result = DATA._searchDirectory(DATA.static,"SDSS_r.xml",errorNotFound=True)
        self.assertTrue(result.endswith("datasets/static/filters/SDSS_r.xml"))
        with patch("galacticus.data.recursiveGlob") as mocked_glob:
            mocked_glob.return_value = ["file_a.dat","file_a.dat"]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = DATA._searchDirectory(DATA.static,"SDSS_r.xml")
                self.assertEqual(result,"file_a.dat")
        with self.assertRaises(RuntimeError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result =  DATA._searchDirectory(".","SDSS_r.xml",errorNotFound=True)
        self.assertIsNone(DATA._searchDirectory(".","SDSS_r.xml",errorNotFound=False))
        return

    def test_GalacticusDataSearchStatic(self):        
        DATA = GalacticusData()
        result = DATA.searchStatic("SDSS_r.xml",errorNotFound=True)
        self.assertTrue(result.endswith("datasets/static/filters/SDSS_r.xml"))
        with patch("galacticus.data.recursiveGlob") as mocked_glob:
            mocked_glob.return_value = ["file_a.dat","file_a.dat"]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = DATA.searchStatic("SDSS_r.xml")
                self.assertEqual(result,"file_a.dat")
        with self.assertRaises(RuntimeError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result =  DATA.searchStatic("SDSS_A.xml",errorNotFound=True)
        self.assertIsNone(DATA.searchStatic("SDSS_A.xml",errorNotFound=False))
        return

    def test_GalacticusDataSearchDynamic(self):
        DATA = GalacticusData()
        with patch("galacticus.data.recursiveGlob") as mocked_glob:
            mocked_glob.return_value = ["file_a.dat","file_a.dat"]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = DATA.searchDynamic("SDSS_r.xml")
                self.assertEqual(result,"file_a.dat")
        with self.assertRaises(RuntimeError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result =  DATA.searchDynamic("SDSS_A.xml",errorNotFound=True)
        self.assertIsNone(DATA.searchDynamic("SDSS_A.xml",errorNotFound=False))
        os.mkdir(DATA.dynamic+"/filters")
        copyfile(DATA.static+"/filters/SDSS_r.xml",DATA.dynamic+"/filters/SDSS_r.xml")
        result = DATA.searchDynamic("SDSS_r.xml",errorNotFound=True)
        self.assertTrue(result.endswith("datasets/dynamic/filters/SDSS_r.xml"))
        os.remove(DATA.dynamic+"/filters/SDSS_r.xml")
        os.rmdir(DATA.dynamic+"/filters")
        return

    def test_GalacticusDataSearch(self):
        DATA = GalacticusData()
        with self.assertRaises(RuntimeError):
            DATA.search("SDSS_A.xml")
        self.assertTrue(DATA.search("SDSS_r.xml").endswith("static/filters/SDSS_r.xml"))
        os.mkdir(DATA.dynamic+"/filters")
        copyfile(DATA.static+"/filters/SDSS_r.xml",DATA.dynamic+"/filters/SDSS_r.xml")
        self.assertTrue(DATA.search("SDSS_r.xml").endswith("dynamic/filters/SDSS_r.xml"))
        os.remove(DATA.dynamic+"/filters/SDSS_r.xml")
        os.rmdir(DATA.dynamic+"/filters")
        return


if __name__ == "__main__":
    unittest.main()

