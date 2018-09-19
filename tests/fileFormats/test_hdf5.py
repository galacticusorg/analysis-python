#! /usr/bin/env python

import os,sys
import warnings
import numpy as np
import unittest
import h5py
from galacticus.fileFormats.hdf5 import HDF5 

def buildTestFile(filename):
    f = h5py.File(filename,'w')
    f.create_group("/Data/ExampleGroup")
    f.create_group("/Header")
    g = f["/Data"]
    attrib = g.attrs
    attrib.create("greeting","hello world",shape=None,dtype=None)
    g.create_dataset("ExampleFloatData",data=np.arange(100,dtype=float),\
                         maxshape=[(None)],\
                         chunks=True,compression="gzip",\
                         compression_opts=6)
    g.create_dataset("ExampleIntData",data=np.arange(100,dtype=int),\
                         maxshape=[(None)],\
                         chunks=True,compression="gzip",\
                         compression_opts=6)
    g = f["/Data/ExampleGroup"]
    g.create_dataset("ExampleFloatData2",data=np.arange(10,dtype=float),\
                         maxshape=[(None)],\
                         chunks=True,compression="gzip",\
                         compression_opts=6)
    g.create_dataset("ExampleIntData2",data=np.arange(10,dtype=int),\
                         maxshape=[(None)],\
                         chunks=True,compression="gzip",\
                         compression_opts=6)
    attrib = f["/Data/ExampleGroup/ExampleIntData2"].attrs
    attrib.create("value",50,shape=None,dtype=None)
    attrib.create("array",np.arange(5,dtype=float),shape=None,dtype=None)
    f.close()
    return


class TestHDF5(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tmpfile = "unitTest1.hdf5"
        self.examplefile = "unitTest2.hdf5"
        buildTestFile(self.examplefile)
        return

    @classmethod
    def tearDownClass(self):
        os.remove(self.tmpfile)
        os.remove(self.examplefile)
        return

    def test_HDF5CreateGroups(self):
        F = HDF5(self.tmpfile,'w')
        F.mkGroup("/Header")
        F.mkGroup("/Data/SubGroup")
        grps0 = F.fileObj["/"].keys()
        self.assertTrue("Header" in grps0)
        self.assertTrue("Data" in grps0)
        grps1 = F.fileObj["/Data"].keys()
        self.assertTrue("SubGroup" in grps1)
        F.close()
        return
    
    def test_HDF5ListGroups(self):
        F = HDF5(self.examplefile,'r')
        grps0 = F.fileObj["/"].keys()
        grps = F.lsGroups("/",recursive=False)
        self.assertTrue(grps == grps0)
        self.assertTrue("Header" in grps)
        self.assertTrue("Data" in grps)
        grps = F.lsGroups("/",recursive=True)
        self.assertEqual(len(grps),3)
        self.assertTrue("/Header" in grps)
        self.assertTrue("/Data" in grps)
        self.assertTrue("/Data/ExampleGroup" in grps)
        grps = F.lsGroups("/Data",recursive=True)
        self.assertEqual(len(grps),1)
        self.assertTrue("/Data/ExampleGroup" in grps)
        grps = F.lsGroups("/Data",recursive=False)
        self.assertEqual(len(grps),1)
        self.assertTrue("ExampleGroup" in grps)
        F.close()
        return

    def test_HDF5ListDatasets(self):
        F = HDF5(self.examplefile,'r')
        self.assertEqual(F.lsDatasets("/",recursive=False),[])
        dsets = F.lsDatasets("/",recursive=True)
        keys = ['/Data/ExampleFloatData','/Data/ExampleGroup/ExampleFloatData2',\
                    '/Data/ExampleGroup/ExampleIntData2','/Data/ExampleIntData']
        self.assertEqual(len(dsets),len(keys))
        [self.assertTrue(key in dsets) for key in keys]
        dsets = F.lsDatasets("/Data",recursive=False)
        keys = ['ExampleFloatData','ExampleIntData']
        self.assertEqual(len(dsets),len(keys))
        [self.assertTrue(key in dsets) for key in keys]
        F.close()
        return

    def testHDF5ListObjects(self):
        F = HDF5(self.examplefile,'r')
        objs = F.lsObjects("/",recursive=False)
        self.assertEqual(len(objs),2)
        keys = ["Header","Data"]
        [self.assertTrue(key in objs) for key in keys]
        objs = F.lsObjects("/",recursive=True)
        self.assertEqual(len(objs),7)
        keys = ['/Data','/Data/ExampleFloatData','/Data/ExampleGroup',\
                    '/Data/ExampleGroup/ExampleFloatData2',\
                    '/Data/ExampleGroup/ExampleIntData2',\
                    '/Data/ExampleIntData','/Header']
        [self.assertTrue(key in objs) for key in keys]
        F.close()
        return

    def test_HDF5ReadDatasets(self):
        F = HDF5(self.examplefile,'r')
        dset = F.readDataset("/Data/ExampleFloatData",exit_if_missing=False)
        [self.assertEqual(a,b) for a,b in zip(dset,np.arange(100,dtype=float))]
        dset = F.readDataset("/Data/ExampleIntData",exit_if_missing=False)
        [self.assertEqual(a,b) for a,b in zip(dset,np.arange(100,dtype=int))]
        self.assertIsNone(F.readDataset("/Data/ExampleData",exit_if_missing=False))        
        with self.assertRaises(KeyError):
            F.readDataset("/Data/ExampleData",exit_if_missing=True)
        dset = F.readDataset("/Data/ExampleGroup/ExampleFloatData2",exit_if_missing=False)
        [self.assertEqual(a,b) for a,b in zip(dset,np.arange(10,dtype=float))]
        dset = F.readDataset("/Data/ExampleGroup/ExampleIntData2",exit_if_missing=False)
        [self.assertEqual(a,b) for a,b in zip(dset,np.arange(100,dtype=int))]
        self.assertIsNone(F.readDataset("/Data/ExampleGroup/ExampleData",exit_if_missing=False))
        with self.assertRaises(KeyError):
            F.readDataset("/Data/ExampleGroup/ExampleData",exit_if_missing=True)
            F.readDataset("/Data/ExampleGroup1/ExampleFloatData2",exit_if_missing=False)
            F.readDataset("/Data/ExampleGroup1/ExampleFloatData2",exit_if_missing=True)
        F.close()
        return

    def test_HDF5ReadAttributes(self):
        F = HDF5(self.examplefile,'r')
        attr = F.readAttributes("/Data")
        self.assertEqual(len(attr.keys()),1)
        self.assertTrue("greeting" in attr.keys())
        self.assertTrue(attr["greeting"],"hello world")
        attr = F.readAttributes("/Data/ExampleGroup/ExampleIntData2")
        self.assertEqual(len(attr.keys()),2)
        self.assertTrue("value" in attr.keys())
        self.assertTrue(attr["value"],50)
        self.assertTrue("array" in attr.keys())
        [self.assertEqual(a,b) for a,b in zip(np.arange(5,dtype=float),attr["array"])]
        attr = F.readAttributes("/Data/ExampleGroup/ExampleIntData2",required=["value"])
        self.assertEqual(len(attr.keys()),1)
        self.assertTrue("value" in attr.keys())
        self.assertTrue(attr["value"],50)
        self.assertFalse("array" in attr.keys())
        with self.assertRaises(KeyError):
            F.readAttributes("/Data/Example")
        F.close()
        return

    def test_HDF5WriteDatasets(self):
        F = HDF5(self.tmpfile,'w')
        F.mkGroup("/Header")
        F.mkGroup("/Data/ExampleGroup")
        # Test writing dataset to root
        data1 = np.random.rand(50)
        F.writeDataset("/Data","ExampleData1",data1)
        self.assertTrue("ExampleData1" in F.fileObj["/Data"].keys())
        self.assertTrue(F.datasetExists("/Data","ExampleData1"))
        diff = np.fabs(data1-np.array(F.fileObj["/Data/ExampleData1"]))
        [self.assertEqual(d,0.0) for d in diff]
        # Test writing dataset to subgroup
        data2 = np.random.rand(50)
        F.writeDataset("/Data/ExampleGroup","ExampleData2",data2)
        self.assertTrue("ExampleData2" in F.fileObj["/Data/ExampleGroup"].keys())
        self.assertTrue(F.datasetExists("/Data/ExampleGroup","ExampleData2"))
        diff = np.fabs(data2-np.array(F.fileObj["/Data/ExampleGroup/ExampleData2"]))
        [self.assertEqual(d,0.0) for d in diff]
        # Test overwriting option
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            F.writeDataset("/Data","ExampleData1",data2,overwrite=False)
            diff = np.fabs(data1-np.array(F.fileObj["/Data/ExampleData1"]))
            [self.assertEqual(d,0.0) for d in diff]
        F.writeDataset("/Data","ExampleData1",data2,overwrite=True)
        diff = np.fabs(data2-np.array(F.fileObj["/Data/ExampleData1"]))
        [self.assertEqual(d,0.0) for d in diff]
        # Testing appending datasets
        data3 = np.random.rand(50)
        F.appendDataset("/Data","ExampleData1",data3)
        self.assertEqual(F.fileObj["/Data/ExampleData1"].size,100)
        diff = np.fabs(np.append(data2,data3)-np.array(F.fileObj["/Data/ExampleData1"]))
        [self.assertEqual(d,0.0) for d in diff]
        with self.assertRaises(KeyError):
            F.appendDataset("/Data","ExampleData3",data3,exit_if_missing=True)
        F.close()
        # Testing writing to file opened in read-only mode
        F = HDF5(self.tmpfile,'r')
        with self.assertRaises(IOError):
            F.writeDataset("/Data","ExampleData1",data1)
        F.close()
        return

    def test_HDF5AddAttributes(self):
        F = HDF5(self.tmpfile,'w')
        F.mkGroup("/Header")
        F.mkGroup("/Data/ExampleGroup")
        data1 = np.random.rand(50)
        F.writeDataset("/Data","ExampleData1",data1)
        attr = {"attr1":"hello world"}
        F.addAttributes("/Data",attr,overwrite=False)
        self.assertEqual(F.fileObj["/Data"].attrs["attr1"],"hello world")
        F.addAttributes("/Data/ExampleData1",{"value":1},overwrite=False)
        self.assertEqual(F.fileObj["/Data/ExampleData1"].attrs["value"],1)
        F.addAttributes("/Data/ExampleData1",{"value":2},overwrite=True)
        self.assertEqual(F.fileObj["/Data/ExampleData1"].attrs["value"],2)
        F.close()
        # Test attempting to write in read-only mode
        F = HDF5(self.tmpfile,'r')
        with self.assertRaises(IOError):
            F.addAttributes("/Data",attr,overwrite=False)
        F.close()
        return

    def test_HDF5RemoveGroups(self):
        F = HDF5(self.tmpfile,'w')
        F.mkGroup("/Header")
        F.mkGroup("/Data/SubGroup")
        grps0 = F.fileObj["/"].keys()
        self.assertTrue("Header" in grps0)
        self.assertTrue("Data" in grps0)
        F.rmGroup("/Header")
        grps0 = F.fileObj["/"].keys()
        self.assertFalse("Header" in grps0)
        self.assertTrue("Data" in grps0)
        F.rmGroup("/Data")
        self.assertFalse("Data" in F.fileObj["/"].keys())
        F.close()        
        F = HDF5(self.tmpfile,'r')
        with self.assertRaises(IOError):
            F.rmGroup("/Header")
        F.close()
        return

    def test_HDF5RemoveDatasets(self):
        F = HDF5(self.tmpfile,'w')
        F.mkGroup("/Header")
        F.mkGroup("/Data/ExampleGroup")
        data1 = np.random.rand(50)
        F.writeDataset("/Data","ExampleData1",data1)
        self.assertTrue("ExampleData1" in F.fileObj["/Data"].keys())
        F.rmDataset("/Data","ExampleData1")
        self.assertFalse("ExampleData1" in F.fileObj["/Data"].keys())
        data2 = np.random.rand(50)
        F.writeDataset("/Data/ExampleGroup","ExampleData2",data2)
        self.assertTrue("ExampleData2" in F.fileObj["/Data/ExampleGroup"].keys())
        F.rmDataset("/Data/ExampleGroup","ExampleData2")
        self.assertFalse("ExampleData2" in F.fileObj["/Data/ExampleGroup"].keys())
        F.close()
        F = HDF5(self.tmpfile,'r')
        with self.assertRaises(IOError):
            F.rmDataset("/Data/ExampleGroup","ExampleData2")
        F.close()
        return
    
    def test_HDF5RemoveAttributes(self):
        F = HDF5(self.tmpfile,'w')
        F.mkGroup("/Header")
        F.mkGroup("/Data/ExampleGroup")
        data1 = np.random.rand(50)
        F.writeDataset("/Data","ExampleData1",data1)
        attr = {"attr1":"hello world","attr2":"foo"}
        F.addAttributes("/Data",attr,overwrite=False)
        self.assertTrue("attr1" in F.fileObj["/Data"].attrs.keys())
        self.assertTrue("attr2" in F.fileObj["/Data"].attrs.keys())
        attr = {"attr1":"goodbye world","attr2":np.random.rand(5)}
        F.addAttributes("/Data/ExampleData1",attr,overwrite=False)
        # Test removing attributes
        F.rmAttributes("/Data",attributes=None)
        self.assertFalse("attr1" in F.fileObj["/Data"].attrs.keys())
        self.assertFalse("attr2" in F.fileObj["/Data"].attrs.keys())
        F.rmAttributes("/Data/ExampleData1",attributes=["attr1"])
        self.assertFalse("attr1" in F.fileObj["/Data/ExampleData1"].attrs.keys())
        self.assertTrue("attr2" in F.fileObj["/Data/ExampleData1"].attrs.keys())
        F.close()
        F = HDF5(self.tmpfile,'r')
        with self.assertRaises(IOError):
            F.rmAttributes("/Data/ExampleData1",attributes=["attr1"])
        F.close()
        return



if __name__ == "__main__":
    unittest.main()
