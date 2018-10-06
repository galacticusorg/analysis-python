#! /usr/bin/env python

import numpy as np
import unittest
import warnings
from scipy.interpolate import interp1d
from galacticus.simulations import locateSimulationSpecsFile
from galacticus.simulations import SimulationBox
from galacticus.simulations import Simulation

class TestLocateSimulationSpecsFile(unittest.TestCase):

    def test_locateSimulationSpecsFile(self):
        simulation = "millennium"
        path = locateSimulationSpecsFile(simulation)
        self.assertTrue(path.endswith("datasets/static/simulations/"+
                                      simulation+".xml"))
        with self.assertRaises(IOError):
            path = locateSimulationSpecsFile("noSuchSimulation")                        
        return


class TestSimulationBox(unittest.TestCase):
    
    def setUp(self):
        self.size = np.array([10.0,15.0,5.0])
        self.PERIODIC = SimulationBox(self.size,periodic=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.NONPERIODIC = SimulationBox(self.size,periodic=False)        
        N = 100
        self.X = np.random.rand(N)*20.0 - 5.0
        self.Y = np.random.rand(N)*25.0 - 5.0
        self.Z = np.random.rand(N)*15.0 - 5.0
        return

    def test_SimulationBox_wrap_dimension(self):
        X1 = self.PERIODIC._wrap_dimension(0,self.X)
        [self.assertTrue(x0,x1) for x0,x1 in zip(self.X%self.size[0],X1)]
        Y1 = self.PERIODIC._wrap_dimension(1,self.Y)
        [self.assertTrue(x0,x1) for x0,x1 in zip(self.Y%self.size[1],Y1)]
        Z1 = self.PERIODIC._wrap_dimension(2,self.Z)
        [self.assertTrue(x0,x1) for x0,x1 in zip(self.Z%self.size[2],Z1)]        
        return
        
    def test_SimulationBoxWrap(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            X1,Y1,Z1 = self.NONPERIODIC.wrap(self.X,self.Y,self.Z)
            [self.assertEqual(x0,x1) for x0,x1 in zip(self.X,X1)]
            [self.assertEqual(y0,y1) for y0,y1 in zip(self.Y,Y1)]
            [self.assertEqual(z0,z1) for z0,z1 in zip(self.Z,Z1)]
        X1,Y1,Z1 = self.PERIODIC.wrap(self.X,self.Y,self.Z)
        [self.assertTrue(x0,x1) for x0,x1 in zip(self.X%self.size[0],X1)]
        [self.assertTrue(x0,x1) for x0,x1 in zip(self.Y%self.size[1],Y1)]
        [self.assertTrue(x0,x1) for x0,x1 in zip(self.Z%self.size[2],Z1)]        
        return
    
class TestSimulation(unittest.TestCase):
    
    def test_Simulation__init__(self):        
        with self.assertRaises(IOError):
            SIM = Simulation("noSuchSimulation")
        SIM = Simulation("millennium")
        self.assertEqual(SIM.name,"Millennium")
        self.assertEqual(SIM.omega0,0.25)
        self.assertEqual(SIM.lambda0,0.75)
        self.assertEqual(SIM.omegaB,0.045)
        self.assertEqual(SIM.h0,0.73)
        self.assertEqual(SIM.H0,73.0)        
        self.assertEqual(SIM.sigma8,0.9)        
        self.assertEqual(SIM.ns,1.0)
        self.assertEqual(SIM.temperatureCMB,2.726)
        [self.assertEqual(SIM.box.size[i],500.0) for i in range(3)]        
        self.assertEqual(SIM.box.units,"Mpc/h")
        self.assertEqual(SIM.box.periodic,True)
        self.assertEqual(SIM.particles.number,10077696000)
        self.assertEqual(SIM.particles.mass,8.606567e8)
        self.assertEqual(SIM.particles.units,"Msol/h")
        self.assertEqual(len(SIM.snapshots.index),63-6)
        self.assertEqual(SIM.snapshots.index[0],7)
        self.assertEqual(SIM.snapshots.z[0],15.343074)
        self.assertEqual(SIM.snapshots.index[-1],63)
        self.assertEqual(SIM.snapshots.z[-1],0.0)
        return
    
    def test_SimulationRedshift(self):
        SIM = Simulation("millennium")            
        redshifts = SIM.redshift(SIM.snapshots.index)
        [self.assertEqual(redshifts[i],SIM.snapshots.z[i]) \
             for i in range(len(redshifts))]
        self.assertEqual(SIM.redshift(63),0.0)
        self.assertTrue(np.isnan(SIM.redshift(72,excludeOutOfBounds=True)))
        self.assertEqual(SIM.redshift(72,excludeOutOfBounds=False),0.0)
        self.assertTrue(np.isnan(SIM.redshift(0,excludeOutOfBounds=True)))
        self.assertEqual(SIM.redshift(0,excludeOutOfBounds=False),15.343074)
        return

    def test_SimulationSnapshot(self):
        SIM = Simulation("millennium")
        N = 100
        redshifts = np.random.rand(N)*30 - 5.0
        # Test case for excluding values out of bounds
        fill = -999
        I = interp1d(SIM.snapshots.z[::-1],SIM.snapshots.index[::-1],
                     bounds_error=False,fill_value=fill)
        index = np.rint(I(redshifts)).astype(int)
        zsnap = SIM.redshift(index,excludeOutOfBounds=True)
        result = SIM.snapshot(redshifts,excludeOutOfBounds=True)
        self.assertEqual(type(result),np.ndarray)
        [self.assertEqual(r,i) for r,i in zip(result,index)]        
        result = SIM.snapshot(redshifts,return_redshift=True,excludeOutOfBounds=True)
        self.assertEqual(type(result),tuple)
        [self.assertEqual(r,i) for r,i in zip(result[0],index)]       
        [self.assertEqual(np.isnan(r),np.isnan(z)) for r,z in zip(result[1],zsnap)]        
        [self.assertEqual(r,z) for r,z in zip(result[1],zsnap) if not np.isnan(r)]        
        # Test case for not excluding values out of bounds
        fill = (SIM.snapshots.index.max(),SIM.snapshots.index.min())
        I = interp1d(SIM.snapshots.z[::-1],SIM.snapshots.index[::-1],
                     bounds_error=False,fill_value=fill)
        index = np.rint(I(redshifts)).astype(int)
        zsnap = SIM.redshift(index,excludeOutOfBounds=False)
        result = SIM.snapshot(redshifts,excludeOutOfBounds=False)
        self.assertEqual(type(result),np.ndarray)
        [self.assertEqual(r,i) for r,i in zip(result,index)]        
        result = SIM.snapshot(redshifts,return_redshift=True,excludeOutOfBounds=False)
        self.assertEqual(type(result),tuple)
        [self.assertEqual(r,i) for r,i in zip(result[0],index)]       
        [self.assertEqual(np.isnan(r),np.isnan(z)) for r,z in zip(result[1],zsnap)]        
        [self.assertEqual(r,z) for r,z in zip(result[1],zsnap) if not np.isnan(r)]        
        return
    
if __name__ == "__main__":
    unittest.main()


