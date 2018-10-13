#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import copy
import warnings
import unittest
from galacticus.data import GalacticusData
from galacticus.parameters import GalacticusParameters
from galacticus.parameters.compare import ParametersMatch


class TestParametersMatch(unittest.TestCase):
    
    def test_ParametersMatchMatchParameter(self):        
        for a in [1,"hello",10.0,np.random.rand(5),
                  np.random.rand(50).reshape(5,10),(2,"a"),None]:
            self.assertTrue(ParametersMatch.matchParameter(a,a))
        self.assertFalse(ParametersMatch.matchParameter(1,0))
        self.assertFalse(ParametersMatch.matchParameter("a","b"))
        r = np.random.rand(5)
        s = np.random.rand(5)
        self.assertFalse(ParametersMatch.matchParameter(r,s))
        r = np.random.rand(5)
        s = np.random.rand(10)
        self.assertFalse(ParametersMatch.matchParameter(r,s))
        r = np.random.rand(50).reshape(5,10)
        s = np.random.rand(100).reshape(10,10)
        self.assertFalse(ParametersMatch.matchParameter(r,s))
        return
        
    def test_ParametersMatchMatch(self):
        DATA = GalacticusData()
        paramfile = DATA.search("snapshotExample.xml")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            PARAMS1 = GalacticusParameters(file=paramfile)
            PARAMS1.mapTree()
            self.assertTrue(ParametersMatch.match(PARAMS1,PARAMS1))
            PARAMS2 = GalacticusParameters(file=paramfile)
            path = "/parameters/mergerTreeConstructorMethod/fileNames"
            PARAMS2.setParameter(path,"/home/Galacticus")
            self.assertTrue(ParametersMatch.match(PARAMS1,PARAMS2))
            path = "/parameters/cosmologyParametersMethod/HubbleConstant"
            PARAMS2.setParameter(path,9999.9)
            self.assertFalse(ParametersMatch.match(PARAMS1,PARAMS2))
        return


if __name__ == "__main__":
    unittest.main()
