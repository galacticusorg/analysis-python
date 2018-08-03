#! /usr/bin/env python

import os,sys,fnmatch,glob,shutil
import numpy as np
from ..fileFormats.hdf5 import HDF5
from ..utils.progress import Progress
from ..datasets import Dataset




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
     getDataType(): Return datatype of specified dataset.
     getMergerTreeWeight(): Return the list of weights to apply to each galaxy.
     getOutputName(): Return name of output nearest to specified redshift.
     getOutputRedshift(): Return redshift of specified output.
     getRedshift(): Return a Dataset class instance containing either lightcone
                    or snapshot redshift information.
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
                ngals = OUT["nodeData/"+dataset].size
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

    
    def getDataset(self,datasetName,z):
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
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        DATA = Dataset()
        DATA.name = datasetName
        if not self.datasetExists(datasetName,z):
            return DATA
        path = "/Outputs/"+self.getOutputName(z)+"/nodeData/"+DATA.name
        DATA.attr = self.readAttributes(path)
        DATA.data = np.array(self.fileObj[path])
        return DATA

    def getDataType(self,datasetName,z):
        """
        GalacticusHDF5.getDataType(): Return the datatype of a dataset. If the dataset does not
                                      exist, then returns 'None'.
                                      
        USAGE:  dtype = GalacticusHDF5.getDataType(datasetName,z)

           INPUTS
               datasetName -- Name of dataset to search for.
               z           -- Redshift to query outputs.
               
           OUTPUTS
               dtype       -- String containing datatype of dataset, or 'None' if dataset
                              is not found.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not self.datasetExists(datasetName,z): 
            return None
        output = self.getOutputName(z)
        dset = self.fileObj["Outputs/"+output+"/nodeData/"+datasetName]
        return str(dset.dtype)
    

    def getMergerTreeWeight(self,z):
        """
        GalacticusHDF5.getMergerTreeWeight(): Return the merger tree weighting to apply to
                                              each galaxy. 

        USAGE:  DATA = GalacticusHDF5.getMergerTreeWeight(z)

           INPUT
               z   -- Redshift to query outputs.

           OUTPUT
              DATA -- Dataset() class object containing weight in DATA.data.
                                              
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        out = self.selectOutput(z)
        cts = np.array(out["mergerTreeCount"])
        wgt = np.array(out["mergerTreeWeight"])        
        DATA = Dataset()
        DATA.name = "mergerTreeWeight"
        DATA.path = "Outputs/"+self.getOutputName(z)+"nodeData/"
        DATA.data = np.copy(np.repeat(wgt,cts))
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


    def getRedshift(self,z):
        """
        GalacticusHDF5.getRedshift(): Return a Dataset class object containing the redshift of 
                                      the galaxies. These either correspond to the lightcone
                                      redshift of the galaxy (if the dataset 'lightconeRedshift'
                                      is present) or the redshift of the snapshot in which the 
                                      galaxies are found.
                                      
        USAGE: DATA = GalacticusHDF5.getRedshift(z)  

              INTPUT
                  z    -- Redshift to query outputs.

              OUTPUT
                  DATA -- A Dataset() class object with redshifts stored in DATA.data.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.datasetExists("lightconeRedshift",z):
            DATA = self.getDataset("lightconeRedshift",z)
        else:
            DATA = Dataset()
            DATA.name = "snapshotRedshift"
            DATA.path = "Outputs/"+self.getOutputName(z)+"nodeData/"
            DATA.data = np.ones(self.countGalaxiesAtRedshift(z),dtype=float)*self.nearestRedshift(z)
        return DATA


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

