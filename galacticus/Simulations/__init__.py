#! /usr/bin/env python

import sys,os,fnmatch,glob
import numpy as np
import warnings
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
from .. import checkDatasetsPath,DATASETS_PATH


def locateSimulationParameterFile(simulation):
    checkDatasetsPath()
    if not os.path.exists(DATASETS_PATH+"static/simulations"):
        msg = "ERROR! Sub-directory 'simulations' not found in datasets."
        raise RuntimeError(msg)
    simulationFile = DATASETS_PATH+"static/simulations/"+simulation.lower()+".xml"
    if not os.path.exists(simulationFile):        
        simulations = glob.glob(DATASETS_PATH+"static/simulations/*.xml")
        simulations = [simfile.split("/")[-1].replace(".xml","") for simfile in simulations]
        msg = "ERROR! Unable to locate simulation parameters file. Simulations available "+\
            "include: "+", ".join(simulations)+"."
        raise IOError(msg)
    return simulationFile


class SimulationBox(object):

    def __init__(self,size,units=None,periodic=True):
        self.size = size
        self.units = units
        self.periodic = periodic
        return

    def _wrap_dimension(self,i,x):
        mask = x < 0.0
        np.place(x,mask,self.size[i]+x[mask])
        mask = x > self.size[i]
        np.place(x,mask,x[mask]-self.size[i])
        return x
        
    def wrap(self,x,y,z):
        if not self.periodic:
            warnings.warn("Cannot wrap positions. Simulation box is not periodic.")
            return x,y,z
        xnew = self._wrap_dimension(0,np.copy(x))
        ynew = self._wrap_dimension(1,np.copy(y))
        znew = self._wrap_dimension(2,np.copy(z))
        return xnew,ynew,znew

        
class SimulationParticles(object):
    
    def __init__(self,number,mass,units=None):
        self.number = number
        self.mass = mass
        self.units = units
        return


class Simulation(object):
    
    def __init__(self,simulation,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose    
        # Load XML file of simulation specifications
        xmlFile = locateSimulationParameterFile(simulation)
        xmlStruct = ET.parse(xmlFile)
        xmlRoot = xmlStruct.getroot()
        xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p}
        # Set simulation name
        self.name = xmlRoot.attrib["name"]
        # Set cosmology
        cosmologyStruct = xmlRoot.find("cosmology")
        self.omega0 = float(cosmologyStruct.find("OmegaM").text)
        self.lambda0 = float(cosmologyStruct.find("OmegaL").text)
        self.omegaB = float(cosmologyStruct.find("OmegaB").text)
        self.H0 = float(cosmologyStruct.find("H0").text)
        self.h0 = self.H0/100.0
        self.sigma8 = float(cosmologyStruct.find("sigma8").text)
        self.ns = float(cosmologyStruct.find("ns").text)
        try:
            self.temperatureCMB = float(cosmologyStruct.find("temperatureCMB").text)
        except AttributeError:
            self.temperatureCMB = 2.726
        # Set box size and number/mass of particles
        size = float(xmlRoot.find("boxSize").text)
        units = xmlRoot.find("boxSize").attrib["units"]
        self.box = SimulationBox([size,size,size],units=units,periodic=True)        
        particles = xmlRoot.find("particles")
        mass = float(particles.find("mass").text)
        massUnits = particles.find("mass").attrib["units"]
        number = particles.find("number").text
        self.particles = SimulationParticles(number,mass,units=massUnits)
        # Set snapshots and corresponding redshifts
        snapshots = xmlRoot.find("snapshots")
        snapshotData = snapshots.findall("snapshot")
        self.snapshots = np.zeros(len(snapshotData),dtype=[("index",int),("z",float)])
        for i,snap in enumerate(snapshotData):
            self.snapshots["z"][i] = float(snap.text)
            self.snapshots["index"][i] = int(snap.attrib["number"])
        self.snapshots = self.snapshots.view(np.recarray)
        return
    
    def specifications(self):
        ndash = 65
        print("-"*ndash                                                                              )
        print(" SPECIFICATIONS: "+self.name                                                          )
        print("            BOX SIZE        = "+str(self.box.size)+" "+str(self.box.units)            )
        print("            NUM. PARTICLES  = "+str(self.particles.number)                            )
        print("            PARTICLE MASS   = "+str(self.particles.mass)+" "+str(self.particles.units))
        print("            MIN. REDSHIFT   = "+str(self.snapshots.z.min())                           )
        print("       Cosmology:"                                                                    )
        print("            OMEGA_MATTER    = "+str(self.omega0)                                      )
        print("            OMEGA_VACUUM    = "+str(self.lambda0)                                     )
        print("            HUBBLE PARAM.   = "+str(self.h0)                                          )
        print("            OMEGA_BARYON    = "+str(self.omegaB)                                      )
        print("            SIGMA_8         = "+str(self.sigma8)                                      )
        print("            POWER SPEC.IND. = "+str(self.ns)                                          )
        print("-"*ndash                                                                              )
        return

    def redshift(self,snapshot,excludeOutOfBounds=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if np.ndim(snapshot) == 0:
            search = np.array([snapshot])
        else:
            search = np.array(snapshot)
        index = np.searchsorted(self.snapshots.index,search)
        if excludeOutOfBounds:
            redshift = np.ones(len(search))*np.nan
            mask = np.logical_and(search>=self.snapshots.index.min(),search<=self.snapshots.index.max())
            index = np.searchsorted(self.snapshots.index,search[mask])
            z = self.snapshots.z[index]
            np.place(redshift,mask,z)
        else:
            index = np.searchsorted(self.snapshots.index,search)
            np.place(index,index==len(self.snapshots.index),len(self.snapshots.index)-1)
            redshift = self.snapshots.z[index]
        if np.ndim(snapshot) == 0:
            redshift = redshift[0]
        return redshift

    def snapshot(self,z,return_redshift=False,excludeOutOfBounds=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if np.ndim(z) == 0:
            zSearch = np.array([z])
        else:
            zSearch = np.array(z)
        if excludeOutOfBounds:
            fill_value = -999
        else:
            fill_value = (self.snapshots.index.max(),self.snapshots.index.min())
        f = interp1d(self.snapshots.z[::-1],self.snapshots.index[::-1],bounds_error=False,\
                         fill_value=fill_value)
        snapshot = np.rint(f(zSearch)).astype(int)
        if np.ndim(z) == 0:
            snapshot = snapshot[0]
        if return_redshift:
            return snapshot,self.redshift(snapshot,excludeOutOfBounds=excludeOutOfBounds)
        return snapshot


