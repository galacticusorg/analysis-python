#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import warnings
import unittest
import xml.etree.ElementTree as ET
from galacticus.fileFormats.xmlTree import xmlTree,indent


class TestXMLTree(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tmpfile = "xmlTree_test_file.xml"
        cls.exFile = "xmlTree_example_file.xml"
        return
    
    def setUp(self):
        ROOT = ET.Element("root")
        tree = ET.ElementTree(element=ROOT)
        ROOT = tree.getroot()
        ELEM1 = ET.SubElement(ROOT,"elem1",attrib={"value":"1"}).text = "hello world"
        ELEM1 = ROOT.find("elem1")
        ELEM2 = ET.SubElement(ELEM1,"elem2")
        ELEM3 = ET.SubElement(ELEM2,"elem3").text = "goodbye world"
        ELEM3 = ET.SubElement(ELEM2,"elem4")
        indent(ROOT)
        tree.write(self.exFile)
        return

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.tmpfile)
        #os.remove(cls.exFile)
        return

    def test_xmlTreeCreateTree(self):
        TREE = xmlTree(root="root")
        TREE.createElement("/root","elem1",attrib={"value":1},text="hello world",createParents=True)
        self.assertTrue(TREE.elementExists("/root/elem1"))
        matches = TREE.matchPath("/root/elem1")
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0],"/root/elem1")
        with self.assertRaises(RuntimeError):
            TREE.createElement("/root/elem2","elem3",createParents=False)
        return

    def test_ListElements(self):
        TREE = xmlTree(file=self.exFile)
        ELEM1 = TREE.tree.getroot().find("elem1")
        ELEMS = TREE.lsElements(TREE.tree.getroot())
        self.assertEqual(len(ELEMS),1)
        self.assertEqual(ELEMS[0],ELEM1)
        ELEM2 = ELEM1.find("elem2")
        ELEMS = TREE.lsElements(ELEM1)
        self.assertEqual(len(ELEMS),1)
        self.assertEqual(ELEMS[0],ELEM2)
        ELEM3 = ELEM2.find("elem3")
        ELEM4 = ELEM2.find("elem4")
        ELEMS = TREE.lsElements(ELEM2)
        self.assertEqual(len(ELEMS),2)
        self.assertTrue(ELEM3 in ELEMS)
        self.assertTrue(ELEM4 in ELEMS)
        return
        
    def test_xmlTreeReadElements(self):
        TREE = xmlTree(file=self.exFile)
        ELEM = TREE.getElement("/root/elem1")
        self.assertIsNotNone(ELEM)
        self.assertEqual(ELEM.text,"hello world")
        self.assertEqual(ELEM.attrib["value"],"1")
        self.assertTrue(TREE.elementExists("/root/elem1/elem2"))
        self.assertTrue(TREE.elementExists("/root/elem1/elem2/elem3"))        
        ELEM = TREE.getElement("/root/elem1/elem2/elem3")
        self.assertEqual(ELEM.text,"goodbye world")
        return

    def test_xmlTreeElementExists(self):
        TREE = xmlTree(file=self.exFile)
        self.assertTrue(TREE.elementExists("/root/elem1"))
        self.assertTrue(TREE.elementExists("/root/elem1/elem2/elem3"))
        self.assertFalse(TREE.elementExists("/root/elem1/elem5"))
        return

    def test_xmlTreeRemoveElements(self):
        TREE = xmlTree(file=self.exFile)
        TREE.removeElement("/root/elem1/elem2/elem3")
        self.assertTrue(TREE.elementExists("/root/elem1/elem2"))
        self.assertFalse(TREE.elementExists("/root/elem1/elem2/elem3"))
        TREE.removeElement("/root/elem1")
        self.assertFalse(TREE.elementExists("/root/elem1"))
        return

    def test_xmlTreeMapTree(self):
        trueMap = ['/', '/root/elem1/elem2/elem3', '/root/elem1/elem2/elem4', 
                   '/root/elem1/elem2', '/root/elem1', '/root']
        TREE = xmlTree(file=self.exFile)
        TREE.tree.map = None
        TREE.mapTree()
        self.assertEqual(len(TREE.map),len(trueMap))
        [self.assertTrue(obj in TREE.map) for obj in trueMap]
        return

    def test_xmlTreeAddElementToMap(self):
        trueMap = ['/', '/root/elem1/elem2/elem3', '/root/elem1/elem2/elem4', 
                   '/root/elem1/elem2', '/root/elem1', '/root']
        TREE = xmlTree(file=self.exFile)
        ELEM4 = TREE.getElement("/root/elem1/elem2/elem4")
        ELEM5 = ET.SubElement(ELEM4,"elem5")
        TREE.addElementToMap(ELEM5,path="/root/elem1/elem2/elem4")
        self.assertTrue("/root/elem1/elem2/elem4/elem5" in TREE.map)
        return

    def test_xmlTreeMatchPath(self):
        TREE = xmlTree(file=self.exFile)
        path = TREE.matchPath("/root/elem1")
        self.assertEqual(len(path),1)
        self.assertEqual(path[0],"/root/elem1")
        path = TREE.matchPath("/root/elem1/elem2")
        self.assertEqual(len(path),1)
        self.assertEqual(path[0],"/root/elem1/elem2")
        # Test use of wildcards for parent directories
        path = TREE.matchPath("*elem2")
        self.assertEqual(len(path),1)
        self.assertEqual(path[0],"/root/elem1/elem2")
        # Test instances when multiple paths identified
        path = TREE.matchPath("/root/elem1/elem2/elem*",errorOnMultiple=False)
        self.assertEqual(len(path),2)
        paths = ["/root/elem1/elem2/elem3","/root/elem1/elem2/elem4"]
        [self.assertTrue(p in path) for p in paths]
        with self.assertRaises(ValueError):
            path = TREE.matchPath("/root/elem1/elem2/elem*")
        return
    
    def test_xmlTreeUpdateElements(self):
        TREE = xmlTree(file=self.exFile)
        TREE.updateElement("/root/elem1",text="goodbye world",attrib={"value":2,"other":4})
        ELEM = TREE.getElement("/root/elem1")
        self.assertEqual(ELEM.text,"goodbye world")
        self.assertEqual(ELEM.attrib["value"],2)
        self.assertTrue("other" in ELEM.attrib.keys())
        self.assertEqual(ELEM.attrib["other"],4)
        TREE.updateElement("/root/elem1/elem2/elem3",text="goodbye world",attrib={"value":5,"other":"hello"})
        ELEM = TREE.getElement("/root/elem1/elem2/elem3")
        self.assertEqual(ELEM.text,"goodbye world")
        self.assertEqual(ELEM.attrib["value"],5)
        self.assertTrue("other" in ELEM.attrib.keys())
        self.assertEqual(ELEM.attrib["other"],"hello")
        return

    def test_xmlTreeWriteToFile(self):
        TREE = xmlTree(file=self.exFile)
        # Test writing to file
        TREE.writeToFile(self.tmpfile)
        self.assertTrue(os.path.exists(self.tmpfile))
        return

    def test_xmlTreeLoadFromFile(self):
        TREE = xmlTree()
        TREE.loadFromFile(self.exFile)
        self.assertEqual(TREE.tree.getroot().tag,"root")
        self.assertEqual(TREE.tree.getroot().find("elem1").tag,"elem1")
        ELEM = TREE.tree.getroot().find("elem1").find("elem2").find("elem3")
        self.assertEqual(ELEM.tag,"elem3")
        return

if __name__ == "__main__":
    unittest.main()
