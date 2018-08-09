#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import warnings
import unittest
import xml.etree.ElementTree as ET


def formatFile(ifile,ofile=None):
    import shutil
    tmpfile = ifile.replace(".xml","_copy.xml")
    if ofile is not None:
        cmd = "xmllint --format "+ifile+" > "+ofile
    else:
        cmd = "xmllint --format "+ifile+" > "+tmpfile
    os.system(cmd)
    if ofile is None:
        shutil.move(tmpfile,ifile)
    return

class xmlTree(object):

    def __init__(self,root="root",file=None):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        ROOT = ET.Element(root)            
        self.tree = ET.ElementTree(element=ROOT,file=file)
        self.map = None
        return

    def loadFromFile(self,xmlfile):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.tree = ET.parse(xmlfile)
        self.mapTree()
        return

    def lsElements(self,OBJ):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        result = None
        if type(OBJ) is ET.ElementTree:
            result = OBJ.findall(".")
        elif type(OBJ) is ET.Element:
            result = list(OBJ)
        else:
            raise TypeError("Object is not an ElementTree or an Element!")
        return result
        
    def mapTree(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.map = ["/"]
        path = ""
        dummy = [self.addElementToMap(E,path=path) for E in self.lsElements(self.tree)]
        return 

    def addElementToMap(self,ELEM,path="/"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.lsElements(ELEM) > 0:
            dummy = [self.addElementToMap(E,path=path+"/"+ELEM.tag) for E in self.lsElements(ELEM)]
        self.map.append(path+"/"+ELEM.tag)
        return
    
    def matchPath(self,path):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.map is None:
            self.mapTree()
        matches = fnmatch.filter(self.map,path)
        self.reportMultipleMatches(matches)
        return matches

    def elementExists(self,path):
        matches = self.matchPath(path)
        return len(matches)==1
    
    
    def reportMultipleMatches(self,matches):
        if len(matches) > 1:
            print("-"*20)
            msg = "Path matches found in XML tree:"
            msg = msg + "\n   "+"\n   ".join(matches)
            print(msg)
            print("-"*20)
            raise ValueError("Multiple path matches found in XML tree!")
        return
        
    def getElementAttribute(self,path,attrib=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        OBJ = self.getElement(path)
        value = None
        if OBJ is None:
            return value
        if attrib is None:
            return OBJ.attrib
        if attrib in OBJ.attrib.keys():
            value = OBJ.attrib[attrib]
        return value

    def getElement(self,querypath):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        matches = self.matchPath(querypath)
        if len(matches)==0:
            return None
        path = matches[0]
        ROOT = self.tree.getroot()
        path = path.replace("/"+ROOT.tag,"")
        OBJ = ROOT
        levels = path.split("/")[1:]
        if len(levels) > 0:
            for dir in levels:
                OBJ = OBJ.find(dir)
        return OBJ

    def createElement(self,path,name,attrib={},text=None,createParents=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        matches = self.matchPath(path)
        if len(matches) == 0:
            if createParents:          
                parentPath = "/".join(path.split("/")[:-1])
                if parentPath == "":
                    parentPath = "/"
                parentName = path.split("/")[-1]
                self.createElement(parentPath,parentName,createParents=createParents)
            else:
                raise RuntimeError(funcname+"(): Parent path does not exist!")
        PARENT = self.getElement(path)
        ET.SubElement(PARENT,name,attrib=attrib).text = text        
        self.map.append(path+"/"+name)
        return

    def updateElement(self,path,attrib={},text=None,createParents=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        matches = self.matchPath(path)        
        if len(matches) == 0:
            parentPath = "/".join(path.split("/")[:-1])
            if parentPath == "":
                parentPath = "/"
            parentName = path.split("/")[-1]
            self.createElement(parentPath,parentName,createParents=createParents)
        ELEM = self.getElement(path)
        for key in attrib.keys():
            ELEM.attrib[key] = attrib[key]                
        if text is not None:
            ELEM.text = text
        return

    def removeElement(self,path):        
        elementName = path.split("/")[-1]
        parentPath = path.replace("/"+elementName,"")
        PARENT = self.getElement(parentPath)
        ELEM = self.getElement(path)
        PARENT.remove(ELEM)
        self.map.remove(path)
        return
    
    def writeToFile(self,outFile,format=True):
        self.tree.write(outFile)
        if format:
            formatFile(outFile)
        return


class UnitTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.tmpfile = "unitTest.xml"
        return
    
    @classmethod
    def tearDownClass(self):
        os.remove(self.tmpfile)
        return

    def testCreateTree(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        print("UNIT TEST: xmlTree: "+funcname)
        print("Creating xmlTree instance")
        TREE = xmlTree(root="root")
        print("Testing creating elements")
        TREE.createElement("/root","elem1",attrib={"value":1},text="hello world",createParents=True)    
        self.assertTrue(TREE.elementExists("/root/elem1"))    
        matches = TREE.matchPath("/root/elem1")
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0],"/root/elem1")
        self.assertRaises(RuntimeError,TREE.createElement,"/root/elem2","elem3",createParents=False)
        print("TEST COMPLETE")
        print("\n")
        return        
        
    def testReadElements(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        print("UNIT TEST: xmlTree: "+funcname)
        print("Creating xmlTree instance")
        TREE = xmlTree(root="root")
        TREE.createElement("/root","elem1",attrib={"value":1},text="hello world",createParents=True)            
        print("Testing reading elements")
        ELEM = None
        ELEM = TREE.getElement("/root/elem1")
        self.assertIsNotNone(ELEM)
        self.assertEqual(ELEM.text,"hello world")
        self.assertEqual(len(ELEM.attrib.keys()),1)
        self.assertEqual(ELEM.attrib["value"],1)
        TREE.createElement("/root/elem2","elem3",text="goodbye world",createParents=True)
        self.assertTrue(TREE.elementExists("/root/elem2"))                
        self.assertTrue(TREE.elementExists("/root/elem2/elem3"))                
        ELEM = TREE.getElement("/root/elem2/elem3")        
        self.assertEqual(ELEM.text,"goodbye world")
        print("TEST COMPLETE")
        print("\n")
        return        

    def testRemoveElements(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        print("UNIT TEST: xmlTree: "+funcname)
        print("Creating xmlTree instance")
        TREE = xmlTree(root="root")
        TREE.createElement("/root","elem1",attrib={"value":1},text="hello world",createParents=True)            
        self.assertTrue(TREE.elementExists("/root/elem1"))    
        TREE.createElement("/root/elem2","elem3",text="goodbye world",createParents=True)
        self.assertTrue(TREE.elementExists("/root/elem2/elem3"))    
        print("Testing removal of elements")
        TREE.removeElement("/root/elem2/elem3")
        self.assertTrue(TREE.elementExists("/root/elem2"))   
        self.assertFalse(TREE.elementExists("/root/elem2/elem3"))    
        self.assertTrue(TREE.elementExists("/root/elem1"))    
        TREE.removeElement("/root/elem1")
        self.assertFalse(TREE.elementExists("/root/elem1"))    
        print("TEST COMPLETE")
        print("\n")
        return        

    def testUpdateElements(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        print("UNIT TEST: xmlTree: "+funcname)
        print("Creating xmlTree instance")
        TREE = xmlTree(root="root")
        TREE.createElement("/root","elem1",attrib={"value":1},text="hello world",createParents=True)            
        ELEM = TREE.getElement("/root/elem1")
        self.assertEqual(ELEM.text,"hello world")
        self.assertEqual(ELEM.attrib["value"],1)
        print("Testing updating element attributes")
        TREE.updateElement("/root/elem1",text="goodbye world",attrib={"value":2,"other":4})
        self.assertEqual(ELEM.text,"goodbye world")
        self.assertEqual(ELEM.attrib["value"],2)
        self.assertTrue("other" in ELEM.attrib.keys())
        self.assertEqual(ELEM.attrib["other"],4)
        ELEM2 = TREE.getElement("/root/elem1")
        self.assertEqual(ELEM2.text,"goodbye world")
        self.assertEqual(ELEM2.attrib["value"],2)
        self.assertTrue("other" in ELEM2.attrib.keys())
        self.assertEqual(ELEM2.attrib["other"],4)
        print("TEST COMPLETE")
        print("\n")        
        return

    def testReadWrite(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        print("UNIT TEST: xmlTree: "+funcname)
        print("Creating xmlTree instance")
        TREE = xmlTree(root="root")
        TREE.createElement("/root","elem1",attrib={"value":"1"},text="hello world",createParents=True)           
        TREE.createElement("/root/elem2","elem3",text="goodbye world",createParents=True)
        print("Testing writing to file")
        TREE.writeToFile(self.tmpfile)
        self.assertTrue(os.path.exists(self.tmpfile))
        del TREE        
        print("Testing reading from file")
        TREE2 = xmlTree(file=self.tmpfile)
        self.assertTrue(TREE2.elementExists("/root/elem1"))
        self.assertTrue(TREE2.elementExists("/root/elem2/elem3"))
        ELEM = None
        ELEM = TREE2.getElement("/root/elem1")
        self.assertIsNotNone(ELEM)
        self.assertEqual(ELEM.text,"hello world")
        self.assertEqual(len(ELEM.attrib.keys()),1)
        self.assertEqual(ELEM.attrib["value"],"1")
        ELEM = TREE2.getElement("/root/elem2/elem3")        
        self.assertEqual(ELEM.text,"goodbye world")
        print("TEST COMPLETE")
        print("\n")        
        return







