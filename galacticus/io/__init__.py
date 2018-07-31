#! /usr/bin/env python

import os,sys,fnmatch,glob,shutil
import numpy as np
from numpy.lib import recfunctions
from ..fileFormats.hdf5 import HDF5
from ..utils.progress import Progress
from ..datasets import Dataset


class Galaxies(object):
    
    def __init__(self):
        self.data = None
        self._mask = None
        self.path = None
        return

    def reset(self):
        self.data = None
        self._mask = None
        self.path = None
        return

    def addDataset(self,name,data,dtype=None):        
        if self._mask is None:
            self._mask = np.ones(len(data),dtype=bool)
        if self.data is None:
            self.data = np.zeros(np.sum(self._mask),dtype=[(name,data.dtype)])
            self.data = self.data.view(np.recarray)
            self.data[name] = np.copy(data[self._mask])
        else:            
            self.data = recfunctions.append_fields(self.data,name,asrecarray=True,\
                                                       data=np.copy(data[self._mask]),\
                                                       dtypes=data.dtype,usemask=False)
        return

    def merge(self,data):
        self.data = np.append(self.data,data)
        return




class GalacticusHDF5(HDF5):
    """
    GalacticusHDF5(): Class to read manage reading/writing to Galacticus HDF5 files.

    Base class: HDF5

    Functions:
    
     availableDatasets(): Return list of names of available galaxy datasets.
     countGalaxies(): Count galaxies in HDF5 file.
     countGalaxiesAtRedshift(): Count galaxies at specified redshift output.
     datasetExists(): Checks whether specified dataset exists.
     getDataset(): Return Dataset class instance with data for specified dataaset.
     getOutputName(): Return name of output nearest to specified redshift.
     getOutputRedshift(): Return redshift of specified output.
     getRedshiftString(): Return string of redshift information used for datasets in
                          output nearest to specified redshift.
     nearestRedshift(): Return redshift of snapshot that is neartest to the specified
                        redshift.
     selectOutput(): Return HDF5 group object for output nearest to specified redshift.

    """
    def __init__(self,*args,**kwargs):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Initalise HDF5 class
        super(GalacticusHDF5, self).__init__(*args,**kwargs)
        # Store version information
        self.version = dict(self.fileObj["Version"].attrs)        
        # Store build information if available
        self.build = None
        if "build" in self.lsGroups("/"):
            self.build = dict(self.fileObj["Build"].attrs)
        # Store parameters
        self.parameters = dict(self.fileObj["Parameters"].attrs)
        self.parameters_parents = { k:"parameters" for k in self.fileObj["Parameters"].attrs.keys()}
        for k in self.fileObj["Parameters"]:
            if len(self.fileObj["Parameters/"+k].attrs.keys())>0:
                d = dict(self.fileObj["Parameters/"+k].attrs)                
                self.parameters.update(d)
                d = { a:k for a in self.fileObj["Parameters/"+k].attrs.keys()}
                self.parameters_parents.update(d)
        # Store output epochs
        self.outputs = None
        if "Outputs" in self.fileObj.keys():
            Outputs = self.fileObj["Outputs"]
            outputKeys = fnmatch.filter(Outputs.keys(),"Output*")
            nout = len(outputKeys)
            if nout > 0:
                isort = np.argsort(np.array([ int(key.replace("Output","")) for key in outputKeys]))
                self.outputs = np.zeros(nout,dtype=[("iout",int),("a",float),("z",float),("name","|S10")])
                for i,out in enumerate(np.array(Outputs.keys())[isort]):
                    self.outputs["name"][i] = out
                    self.outputs["iout"][i] = int(out.replace("\n","").replace("Output",""))
                    a = float(Outputs[out].attrs["outputExpansionFactor"])
                    self.outputs["a"][i] = a
                    self.outputs["z"][i] = (1.0/a) - 1.0
                self.outputs = self.outputs.view(np.recarray)
        return

    def availableDatasets(self,z):
        """
        GalacticusHDF5.availableDatasets(): Returns a list of galaxy properties available in the 
                                            snapshot output that is closest to specified redshift.
                             
        USAGE:  datasets = GalacticusHDF5.availableDatasets(z)

              INPUTS
                      z  -- redshift value to query. [Default=None]

              OUTPUTS
                datasets -- List of datasets available in this output.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        out = self.selectOutput(z)
        if out is None:
            return []
        return map(str,out["nodeData"].keys())


    def countGalaxies(self,z=None):
        """
        countGalaxies(): Count number of galaxies stored either in the Galacticus HDF5, or just in
                         a single redshift snapshot (seaerches for nearest snapshot closest in
                         redshift).

        USAGE:   ngals = GalacticusHDF5.countGalaxies([z])

                Inputs:
                      z   : redshift value to query (default = None)
               Outputs:
                    ngals : integer count of number of galaxies
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.outputs is None:
            return 0
        if z is None:
            redshifts = self.outputs.z
        else:
            redshifts = [z]
        galaxies = np.array([self.countGalaxiesAtRedshift(redshift) for redshift in redshifts])
        return np.sum(galaxies)

    def countGalaxiesAtRedshift(self,z):
        """
        GalacticusHDF5.countGalaxiesAtRedshift(): Count number of galaxies stored in output that 
                                                  is nearest to the specified redshift.

        USAGE:  ngals = GalacticusHDF5.countGalaxiesAtRedshift(z)

                INPUT
                      z   -- Redshift value to query.
               OUTPUT
                    ngals -- Integer count of number of galaxies.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        ngals = 0
        OUT = self.selectOutput(z)
        if OUT is None:
            return ngals
        if "nodeData" in OUT.keys():            
            if len(self.availableDatasets(z)) > 0:
                dataset = self.availableDatasets(z)[0]            
                ngals = len(np.array(OUT["nodeData/"+dataset]))
        return ngals

    def datasetExists(self,datasetName,z):
        """
        GalacticusHDF5.datasetExists(): Query whether specified dataset exists in the output that
                                        is closest to the specified redshift.

        USAGE: exists = GalacticusHDF5.datasetExists(datasetName,z)

                INPUTS
                   datasetName -- Name of dataset to search.
                      z        -- Redshift to query.
               OUTPUTS
                    exists     -- Logical indicating whether specified dataset is present.
               
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return len(fnmatch.filter(self.availableDatasets(z),datasetName))>0

    
    def getDataset(self,datasetName,z,unitsInSI=None):
        """
        GalacticusHDF5.getDataset(): Extract data for specified dataset name at specified redshift.
                                     Returns empty Dataset class if dataset not found in HDF5 file.

        USAGE: data = GalacticusHDF5.getDataset(datasetName,z)

                INPUTS
                   datasetName -- Name of dataset to search.
                      z        -- Redshift to query.
               OUTPUTS
                    data       -- Dataset class object (see datasets.Dataset).
               
        """
        DATA = Dataset()
        DATA.name = datasetName
        DATA.path = "Outputs/"+self.getOutputName(z)+"/nodeData/"
        if not self.datasetExists(datasetName,z):
            return DATA
        attr = self.readAttributes(DATA.path+DATA.name)
        if "unitsInSI" in attr.keys():
            DATA.unitsInSI = attr["unitsInSI"]
        else:
            DATA.unitsInSI = unitsInSI
        DATA.data = np.array(self.fileObj[DATA.path+DATA.name])
        return DATA
        

    def getOutputName(self,z):
        """
        GalacticusHDF5.getOutputName(): Return the name (e.g. Output002) of the output that
                                        is closest to the specified redshift.

        USAGE:  name = GalacticusHDF5.getOutputName(z)

            INPUTS   
                 z    -- Redshift to query.
           OUTPUTS
                 name -- Name of the nearest output.

        Note: Will return 'None' if no outputs were stored in the HDF5 file. This happens
              if there are no galaxies stored in that output.            
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.outputs is None:
            return None
        # Select epoch closest to specified redshift
        iselect = np.argmin(np.fabs(self.outputs.z-z))
        return self.outputs.name[iselect]


    def getOutputRedshift(self,outputName):        
        """
        GalacticusHDF5.getOutputRedshift(): Return the redshift corresponding to specified 
                                            output name.

        USAGE:  z = GalacticusHDF5.getOutputRedshift(outputName)
        
               INPUT
                     outputName -- String with name of output (e.g. Output1)
               OUTPUTS
                          z     -- Redshift corresponding to specified output.

        """
        i = int(np.argwhere(self.outputs.name=="Output"+outputName.replace("Output","")))
        return self.outputs.z[i]


    def getRedshiftString(self,z):
        """
        GalacticusHDF5.getRedshiftString(): Return the redshift string that is used in the names 
                                            of the datasets stored in the output that is nearest
                                            to the specified redshift.
                             
        USAGE:  zString = GalacticusHDF5.getRedshiftString(z)

                INTPUTS  
                      z     -- Redshift to query.
               OUTPUTS
                    zString -- String containing redshift information used in dataset names.

        For example, for a snapshot with redshift z = 1.4, this function would return a string similar to
        'z1.400', which could then be used to query/construct dataset names.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return fnmatch.filter(fnmatch.filter(self.availableDatasets(z),"*z[0-9].[0-9]*")[0].split(":"),"z*")[0]
    
    def getUUID(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        keys = list(map(str,self.fileObj["/"].attrs.keys()))
        uuid = None
        if "UUID" in keys:            
            uuid = str(self.fileObj["/"].attrs["UUID"])
        return uuid

    def globalHistory(self,props=None,si=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        globalHistory = self.fileObj["globalHistory"]
        allprops = globalHistory.keys() + ["historyRedshift"]
        if props is None:
            props = allprops 
        else:
            props = set(props).intersection(allprops)            
        epochs = len(np.array(globalHistory["historyExpansion"]))
        dtype = np.dtype([ (str(p),np.float) for p in props ])
        history = np.zeros(epochs,dtype=dtype)
        if self._verbose:
            if si:
                print("WARNING! "+funcname+"(): Adopting SI units!")
            else:
                print("WARNING! "+funcname+"(): NOT adopting SI units!")        
        for p in history.dtype.names:
            if p is "historyRedshift":
                history[p] = np.copy((1.0/np.array(globalHistory["historyExpansion"]))-1.0)
            else:
                history[p] = np.copy(np.array(globalHistory[p]))
                if si:
                    if "unitsInSI" in globalHistory[p].attrs.keys():
                        unit = globalHistory[p].attrs["unitsInSI"]
                        history[p] = history[p]*unit
        return history.view(np.recarray)

    def nearestRedshift(self,z):
        """
        GalacticusHDF5.nearestRedshift(): Return the redshift of the output that is closest 
                                          to the specified redshift.
                                          
        USAGE:  zout = GalacticusHDF5.nearestRedshift(z)

            INPUTS
                 z    -- Redshift to query.
           OUTPUTS
                 zout -- Redshift of the nearest output.

        Note: Will return 'None' if no outputs were stored in the HDF5 file. This happens
              if there are no galaxies stored in that output.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.outputs is None:
            return None
        # Select epoch closest to specified redshift
        iselect = np.argmin(np.fabs(self.outputs.z-z))
        return self.outputs.z[iselect]

        



    def readGalaxies(self,z,props=None,SIunits=False,removeRedshiftString=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create array to store galaxy data
        self.galaxies = None
        # Read galaxies from one or more outputs
        if np.ndim(z) == 1:
            if len(z) == 1:
                z = z[0]
        if np.ndim(z) == 0:
            self.readGalaxiesAtRedshift(z,props=props,SIunits=SIunits,removeRedshiftString=removeRedshiftString)
        else:
            zout = np.unique([self.outputs.z[np.argmin(np.fabs(self.outputs.z-iz))] for iz in z])
            PROG = Progress(len(zout))
            dummy = [self.readGalaxiesAtRedshift(iz,props=props,SIunits=SIunits,removeRedshiftString=True,progressObj=PROG) for iz in zout]
        return self.galaxies

    
    def readGalaxiesAtRedshift(self,z,props=None,SIunits=False,removeRedshiftString=False,progressObj=None):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Initiate class for snapshot output
        OUTPUT = SnapshotOutput(z,self)
        # Read galaxies from snapshot
        OUTPUT.readGalaxies(props=props,SIunits=SIunits,removeRedshiftString=removeRedshiftString)
        # Add to galaxies data array
        if self.galaxies is None:
            self.galaxies = np.copy(OUTPUT.galaxies)
        else:
            self.galaxies = np.append(self.galaxies,np.copy(OUTPUT.galaxies))
        # Delete output class
        del OUTPUT
        # Report progress
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line(task="Redshift = "+str(z))
        return 
    
    def selectOutput(self,z):
        """
        GalacticusHDF5.selectOutput(): Return an HDF5 group object for the output that is
                                       closest to the specified redshift.
        
        USAGE: output = GalacticusHDF5.selectOutput(z)
        
           INPUTS
               z      -- Redshift to query.
          OUTPUTS
               output -- HDF5 group object for nearest output. 

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.outputs is None:
            return None
        # Select epoch closest to specified redshift        
        iselect = np.argmin(np.fabs(self.outputs.z-z))
        outstr = "Output"+str(self.outputs["iout"][iselect])
        if self._verbose:
            print(funcname+"(): Nearest output is "+outstr+" (redshift = "+str(self.outputs.z[iselect])+")")
        return self.fileObj["Outputs/"+outstr]







      
class SnapshotOutput(object):
    
    def __init__(self,redshift,galHDF5Obj):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Store GalacticusHDF5 object
        self.galHDF5Obj = galHDF5Obj
        # Select redshift and output
        self.redshift = self.galHDF5Obj.nearestRedshift(redshift)
        self.redshiftString = self.galHDF5Obj.getRedshiftString(self.redshift)
        self.out = self.galHDF5Obj.selectOutput(self.redshift)
        # Count number of galaxies
        self.numberGalaxies = self.galHDF5Obj.countGalaxiesAtRedshift(self.redshift)
        return
    

    def getGalaxyDataset(self,datasetName,SIunits=False,dataTypeName=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set name to store galaxy dataset under
        if dataTypeName is None:
            dataTypeName = datasetName
        # Store galaxy data
        if datasetName in self.galHDF5Obj.availableDatasets(self.redshift):
            self.galaxies[dataTypeName] = np.copy(np.array(self.out["nodeData/"+datasetName]))
            if SIunits:
                if "unitsInSI" in out["nodeData/"+p].attrs.keys():
                    unit = out["nodeData/"+p].attrs["unitsInSI"]
                    self.galaxies[dataTypeName] *= unit
        # Special cases!
        if datasetName in special_cases:
            if datasetName in ["weight","mergerTreeWeight"]:
                cts = np.array(self.out["mergerTreeCount"])
                wgt = np.array(self.out["mergerTreeWeight"])
                self.galaxies[dataTypeName] = np.copy(np.repeat(wgt,cts))
                del cts,wgt
            if datasetName == "snapshotRedshift":
                self.galaxies[dataTypeName] = np.ones_like(self.galaxies[dataTypeName])*self.redshift
            if datasetName in ["lightconeRightAscension","lightconeDeclination"]:
                available = list(set(["lightconePositionX","lightconePositionY","lightconePositionZ"]).intersection(self.galHDF5Obj.availableDatasets(self.redshift)))
                if len(available) != 3:
                    print("WARNING! "+funcname+"(): at one of lightconePosition[XYZ] not found -- unable to compute "+datasetName+"!")
                    self.galaxies[dataTypeName] = np.ones_like(self.galaxies[dataTypeName])*999.9
                else:
                    rightAscension,declination = getRaDec(np.array(self.out["nodeData/lightconePositionX"]),np.array(self.out["nodeData/lightconePositionY"]),\
                                                              np.array(self.out["nodeData/lightconePositionZ"]),degrees=True)
                    if datasetName == "lightconeRightAscension":
                        self.galaxies[dataTypeName] = np.copy(rightAscension)
                    if datasetName == "lightconeDeclination":
                        self.galaxies[dataTypeName] = np.copy(declination)
                    del rightAscension,declination
        return


    def _getDataTypeNames(self,prop):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        prop = prop.replace(":"+self.redshiftString,"")
        allpropsZ = self.galHDF5Obj.availableDatasets(self.redshift)
        allprops = [p.replace(":"+self.redshiftString,"") for p in allpropsZ]
        matches = fnmatch.filter(allprops,prop) + fnmatch.filter(allprops,prop) + fnmatch.filter(special_cases,prop)
        matches = [m.replace(":"+self.redshiftString,"") for m in matches]
        matches = list(np.unique(matches))
        dummy = [self._dataTypeNames.append(m) for m in matches]
        return
    
    def _findDatasetName(self,prop):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if prop in special_cases:
            self._datasetNames.append(prop)
            return        
        allpropsZ = self.galHDF5Obj.availableDatasets(self.redshift)
        allprops = [p.replace(":"+self.redshiftString,"") for p in allpropsZ]              
        index = np.where(np.array(allprops)==prop)[0][0]
        self._datasetNames.append(allpropsZ[index])
        return
    
    def getDatasetType(self,datasetName,dataTypeName=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName not in self.galHDF5Obj.availableDatasets(self.redshift):
            raise KeyError(funcname+"(): dataset '"+datasetName+"' not found in file '"+self.galHDF5Obj.fileObj.filename+"'!")
        # Get datatype
        dtype = getDataType(self.out["nodeData/"+datasetName])
        # Select name to use in datatype
        if dataTypeName is None:
            dataTypeName = datasetName
        return (dataTypeName,dtype)
    
    def _buildDataType(self,datasetName,dataTypeName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if datasetName in special_cases:
            self._dtype.append((dataTypeName,float))
            return
        self._dtype.append(self.getDatasetType(datasetName,dataTypeName=dataTypeName))
        return
                                   
    def readGalaxies(self,props=None,SIunits=False,removeRedshiftString=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get list of datasets and corresponding names in file
        self._dataTypeNames = []
        dummy = [self._getDataTypeNames(prop) for prop in props]
        self._datasetNames = []
        dummy = [self._findDatasetName(prop) for prop in self._dataTypeNames]
        if not removeRedshiftString:
            self._dataTypeNames = self._datasetNames
        # Build galaxies array
        self._dtype = []
        dummy = [self._buildDataType(self._datasetNames[i],dataTypeName=self._dataTypeNames[i])\
                     for i in range(len(self._dataTypeNames))]
        self.galaxies = np.zeros(self.numberGalaxies,dtype=self._dtype)
        # Read properties        
        dummy = [self.getGalaxyDataset(self._datasetNames[i],SIunits=False,dataTypeName=self._dataTypeNames[i])\
                     for i in range(len(self._datasetNames))]
        del self._datasetNames,self._dataTypeNames,self._dtype
        return
        

    
    
class checkOutputFiles(object):
    
    def __init__(self,verbose=True):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        self.complete = []
        self.incomplete = []
        self.corrupted = []
        self.notfound = []
        return

    def reset(self):
        self.complete = []
        self.incomplete = []
        self.corrupted = []
        self.notfound = []
        return


    def checkFile(self,hdf5File,PROG=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not os.path.exists(hdf5File):
            self.notfound.append(hdf5File)
        try:
            GH5 = GalacticusHDF5(hdf5File,'r')
        except IOError:
            self.corrupted.append(hdf5File)
        else:
            attrib = GH5.readAttributes("/")
            if "galacticusCompleted" in attrib.keys():
                if bool(attrib["galacticusCompleted"]):
                    self.complete.append(hdf5File)
                else:
                    self.incomplete.append(hdf5File)
            else:
                self.corrupted.append(hdf5File)
            GH5.close()
        if PROG is not None:
            PROG.increment()
            if self.verbose:
                PROG.print_status_line()
        return

    def checkDirectory(self,outdir,prefix="galacticus_*[0-9]"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not os.path.exists(outdir):
            raise IOError(funcname+"(): directory "+outdir+" does not exist!")
        if not outdir.endswith("/"):
            outdir = outdir + "/"
        files = glob.glob(outdir+prefix+".hdf5")
        PROG = None
        if self.verbose:
            print(funcname+"(): checking HDF5 files...")
            PROG = Progress(len(files))
        dummy = [self.checkFile(ofile,PROG=PROG) for ofile in files]
        return

    def checkFiles(self,files):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        PROG = None
        if self.verbose:            
            print(funcname+"(): checking HDF5 files...")
            PROG = Progress(len(files))
        dummy = [self.checkFile(ofile,PROG=PROG) for ofile in files]
        return


    def copyFile(self,ofile,overwrite=False,PROG=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        nfile = ofile.replace(".hdf5",".RAW.hdf5")
        if os.path.exists(nfile) and not overwrite:
            shutil.copy2(ofile,nfile)
        if PROG is not None:
            PROG.increment()
            if self.verbose:
                PROG.print_status_line()
        return

    def copyFiles(self,files,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        PROG = None
        if self.verbose:            
            print(funcname+"(): copying HDF5 files...")
            PROG = Progress(len(files))
        dummy = [self.copyFile(ofile,overwrite=overwrite,PROG=PROG) for ofile in files]
        return

