#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import warnings
import unittest
import xml.etree.ElementTree as ET

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    return

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
        if len(self.lsElements(ELEM)) > 0:
            dummy = [self.addElementToMap(E,path=path+"/"+ELEM.tag) for E in self.lsElements(ELEM)]
        self.map.append(path+"/"+ELEM.tag)
        return
    
    def matchPath(self,path,errorOnMultiple=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.map is None:
            self.mapTree()
        matches = fnmatch.filter(self.map,path)
        if errorOnMultiple:
            self.reportMultipleMatches(matches)
        return matches

    def elementExists(self,path):
        matches = self.matchPath(path)
        return len(matches)==1
    
    def reportMultipleMatches(self,matches):
        if len(matches) > 1:
            msg = "Multiple path matches found in XML tree!"
            msg = msg + " Path matches found in XML tree:  "
            msg = msg + ", ".join(matches)
            raise ValueError(msg)
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
        if format:
            root = self.tree.getroot()            
            indent(root)
        self.tree.write(outFile)
        return


