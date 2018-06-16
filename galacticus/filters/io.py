#! /usr/bin/env python

import sys,os
import numpy as np
import xml.etree.ElementTree as ET
from . import Filter,computeEffectiveWavelength
from ..fileFormat.xmlTree import formatFile
from ..data import GalacticusData

def loadFilterFromFile(filterFile):
    FILTER = Filter()
    FILTER.file = filterFile
    # Open xml file and load structure
    xmlStruct = ET.parse(self.file)
    xmlRoot = xmlStruct.getroot()
    xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p}
    # Read filter transmission
    response = xmlRoot.find("response")
    data = response.findall("datum")
    FILTER.transmission = np.zeros(len(data),dtype=[("wavelength",float),("transmission",float)])
    for i,datum in enumerate(data):
        FILTER.transmission["wavelength"][i] = float(datum.text.split()[0])
        FILTER.transmission["transmission"][i] = float(datum.text.split()[1])
    FILTER.transmission = FILTER.transmission.view(np.recarray)
    # Read header/information
    FILTER.description = xmlRoot.find("description").text
    FILTER.name = xmlRoot.find("name").text
    if "effectiveWavelength" in xmlMap.keys():
        FILTER.effectiveWavelength = float(xmlRoot.find("effectiveWavelength").text)
    else:
        FILTER.setEffectiveWavelength()
    if "vegaOffset" in xmlMap.keys():
        FILTER.vegaOffset = float(xmlRoot.find("vegaOffset").text)
    if "url" in xmlMap.keys():
        FILTER.url = xmlRoot.find("url").text
    if "origin" in xmlMap.keys():
        FILTER.origin = xmlRoot.find("origin").text
    del xmlStruct, xmlRoot,xmlMap
    return FILTER

def writeFilterToFile(FILTER,path):
    # Create tree root
    root = ET.Element("filter")
    # Add name and other descriptions
    if FILTER.name is None:
        raise ValueError(funcname+"(): must provide a name for the filter!")
    ET.SubElement(root,"name").text = FILTER.name
    description = FILTER.description
    if description is None:
        description = FILTER.name
    ET.SubElement(root,"description").text = description
    origin = FILTER.origin
    if origin is None:
        origin = "unknown"
    ET.SubElement(root,"origin").text = origin
    url = FILTER.url
    if url is None:
        url = "unknown"
    ET.SubElement(root,"url").text = url
    # Add in response data                                                                                                                                                  
    if FILTER.transmission is None:
        raise ValueError(funcname+"(): no transmission curve provided for filter!")
    RES = ET.SubElement(root,"response")
    for i in range(len(FILTER.transmission.wavelength)):
        datum = "{0:7.3f} {1:9.7f}".format(FILTER.transmission.wavelength[i],\
                                               FILTER.transmission.transmission[i])
        ET.SubElement(RES,"datum").text = datum
    # Compute effective wavelength and Vega offset if needed    
    if FILTER.effectiveWavelength is None:
        FILTER.setEffectiveWavelength()
    ET.SubElement(root,"effectiveWavelength").text = str(FILTER.effectiveWavelength)
    if FILTER.vegaOffset is not None:
        ET.SubElement(root,"vegaOffset").text = str(FILTER.vegaOffset)
    # Finalise tree and save to file
    tree = ET.ElementTree(root)
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + self.name+".xml"
    if verbose:
        print(funcname+"(): writing filter to file: "+path)
    tree.write(path)
    formatFile(path)
    return
