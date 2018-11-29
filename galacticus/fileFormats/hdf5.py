#! /usr/bin/env python

import six
import sys,os
import h5py
import numpy as np
import fnmatch
import unittest
import warnings
from ..strings import removeByteStrings,addByteStrings

def flattenNestedList(l):
    return [item for sublist in l for item in sublist]

def findMatchingItems(allItems,itemsToFind):
    found = [fnmatch.filter(allItems,item) for item in itemsToFind]
    return list(set(flattenNestedList(found)))

def findMissingItems(allItems,itemsToSearch):
    missing = [len(fnmatch.filter(allItems,item))==0 for item in itemsToSearch]
    return [item for item, miss in zip(itemsToSearch, missing) if miss]

def readonlyWrapper(func):
    """
    Wrapper to check whether HDF5 file has been opened in read-only mode.    
    """
    def wrapper(self,*args,**kwargs):               
        funcname = self.__class__.__name__+"."+func.__name__
        if self.read_only:
            raise IOError(funcname+"(): HDF5 file "+self.filename+" is READ ONLY!")        
        return func(self,*args,**kwargs)
    return wrapper


class HDF5(object):
    """ 
    HDF5: Class for reading/writing HDF5 files.
    
          USAGE: OBJ = HDF5(filename,ioStatus,verbose=<verbose>)
    
          INPUTS 
             filename -- Path to HDF5 file.  
             ioStatus -- Read ('r'), write ('w') or append ('a') to file.  
              verbose -- Print extra information (default value = False).
    
          OUTPUTS
                OBJ  -- HDF5 class object.

    Attributes:
         fileObj: The h5py.File object.
         filename: String containing HDF5 file path.
         read_only: Logical indicating whether file opened in read only mode.
         

    Functions:
         


    """    
    def __init__(self,*args,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.fileObj = h5py.File(*args)
        if "verbose" in kwargs.keys():
            self.verbose = kwargs["verbose"]
        else:
            self.verbose = False
        self.filename = self.fileObj.filename        
        if self.verbose:
            print(classname+"(): HDF5 file = "+self.filename)
        if self.fileObj.mode == "r":
            self.read_only = True
            if self.verbose:
                print(classname+"(): HDF5 opened in READ-ONLY mode")
        elif self.fileObj.mode == "r+":
            self.read_only = False
        return
    
    def close(self):
        """
        HDF5.close(): Close the HDF5 file instance.

        USAGE: HDF5.close()

        """
        self.fileObj.close()
        return

    def lsObjects(self,hdfdir,recursive=False):
        """
        HDF5.lsObjects(): List all of the objects in the specified directory 
                          inside the HDF5 file.
                          
        USAGE:  objs = HDF5.lsObjects(dir,[recursive=<recursive>])
        
             INPUTS
                   dir       -- Path to HDF5 group.
                   recursive -- Recursively search in sub-groups. [Default=False]
                   
            OUTPUTS
                     objs    -- List of object names.
                 
        """
        ls = []
        thisdir = self.fileObj[hdfdir]
        for obj in thisdir.keys():
            if recursive:            
                path = hdfdir+"/"+str(obj)
                path = path.replace("//","/")
                ls.append(path)                    
                if isinstance(thisdir[obj],h5py.Group):                    
                    ls = ls + self.lsObjects(path+"/",recursive=recursive)
            else:
                ls.append(str(obj))
        return ls

    ##############################################################################
    # GROUPS
    ##############################################################################
    
    @readonlyWrapper
    def mkGroup(self,hdfdir,recursive=True):        
        """
        HDF5.mkGroup(): create HDF5 group with specified path.
        
        USAGE:  HDF5.mkdir(dir)

              Input: dir -- path to HDF5 group.       
        """
        if hdfdir not in self.fileObj:            
            if recursive:                
                parent = "/".join(hdfdir.split("/")[:-1])                
                if len(parent.strip())>0:
                    self.mkGroup(parent,recursive=recursive)           
            g = self.fileObj.create_group(hdfdir)            
        return

    
    @readonlyWrapper
    def rmGroup(self,hdfdir):
        """
        HDF5.rmGroup(): remove HDF5 group at specified path.
        
        USAGE:  HDF5.rmdir(dir)

              Input: dir -- path to HDF5 group.       
        """
        if hdfdir in self.fileObj:
            del self.fileObj[hdfdir]
        return


    @readonlyWrapper
    def cpGroup(self,srcfile,srcdir,dstdir=None):        
        """
        HDF5.cpGroup(): copy HDF5 group with specified path from specified file.
        
        USAGE:  HDF5.cpGroup(srcfile,srcdir,[dstdir])

              Input: srcfile -- Path to source HDF5 file.       
                     srcdir -- Path to source HDF5 group inside source file.
                     dstdir -- Path to group to store copy of source group. 
                               Default = srcdir.

              Note that this function will create in the current file a parent 
              group with the same path as the parent group of the source group
              in the source file.
                                              
        """
        # Open second file and get path to group that we want to copy
        fileObj = h5py.File(srcfile,"r")        
        group_path = fileObj[srcdir].parent.name
        # Create same parent group in current file
        group_id = self.fileObj.require_group(group_path)
        # Set name of new group
        if dstdir is None:
            dstdir = srcdir
        fileObj.copy(srcdir,group_id,name=dstdir)
        fileObj.close()   
        return
    
    def lsGroups(self,hdfdir,recursive=False):
        ls = []
        thisdir = self.fileObj[hdfdir]
        for obj in thisdir.keys():
            if isinstance(thisdir[obj],h5py.Group):
                if recursive:
                    path = hdfdir+"/"+str(obj)
                    path = path.replace("//","/")
                    ls.append(path)
                    ls = ls + self.lsGroups(path+"/",recursive=recursive)
                else:
                    ls.append(str(obj))
        return ls


    ##############################################################################
    # DATASETS
    ##############################################################################


    @readonlyWrapper
    def writeDataset(self,hdfdir,name,data,maxshape=tuple([None]),overwrite=False,\
                         chunks=True,compression="gzip",compression_opts=6,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 group
        if hdfdir not in self.fileObj.keys():
            self.mkGroup(hdfdir)
        g = self.fileObj[hdfdir]
        # Check if dataset exists
        if self.datasetExists(hdfdir,name,exit_if_missing=False):
            if not overwrite:
                warnings.warn(funcname+"(): Unable to write dataset (overwrite=False).")
                return
            else:
                del g[name]
        # Write dataset
        dset = g.create_dataset(name,data=data,maxshape=maxshape,\
                                    chunks=chunks,compression=compression,\
                                    compression_opts=compression_opts,**kwargs)
        return

    @readonlyWrapper
    def appendDataset(self,hdfdir,name,data,exit_if_missing=False,\
                          axis=0,maxshape=tuple([None]),chunks=True,\
                          compression="gzip",compression_opts=6,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check if dataset exists (write dataset fif it does not exist)
        if not self.datasetExists(hdfdir,name,exit_if_missing=exit_if_missing):        
            self.writeDataset(hdfdir,name,data,maxshape=maxshape,chunks=chunks,\
                          compression=compression,compression_opts=compression_opts,\
                                  **kwargs)
            return
        # Select dataset
        if axis != 0:
            raise ValueError(funcname+"(): Currently only implemented for axis=0")
        dset = self.fileObj[hdfdir+"/"+name]
        n = dset.shape[axis]
        dset.resize(dset.shape[axis]+data.shape[axis],axis=axis) 
        dset[n:] = np.copy(data)    
        return
        
    @readonlyWrapper
    def addDataset(self,hdfdir,name,data,append=False,overwrite=False,\
                        maxshape=tuple([None]),chunks=True,compression="gzip",\
                        compression_opts=6,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Select HDF5 group
        if hdfdir not in self.fileObj.keys():
            self.mkGroup(hdfdir)
        # Write dataset
        if append:
            # i) Append to existing dataset
            self.appendDataset(hdfdir,name,data,axis=0,maxshape=maxshape,chunks=chunks,\
                                   compression=compression,compression_opts=compression_opts,\
                                   **kwargs)
        else:
            # ii) Write or over-write existing dataset
            self.writeDataset(hdfdir,name,data,maxshape=maxshape,overwrite=overwrite,\
                             chunks=chunks,compression=compression,compression_opts=compression_opts,\
                                  **kwargs)
        return

            
    @readonlyWrapper
    def addDatasets(self,hdfdir,data,append=False,overwrite=False,\
                        maxshape=tuple([None]),chunks=True,compression="gzip",\
                        compression_opts=6,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 group
        if hdfdir not in self.fileObj.keys():
            self.mkGroup(hdfdir)
        # Write data to group
        dummy = [ self.addDataset(hdfdir,n,data[n],append=append,overwrite=overwrite,\
                                      maxshape=maxshape,chunks=chunks,compression=compression,\
                                      compression_opts=compression_opts,**kwargs) \
                      for n in data.dtype.names ]
        del dummy
        return

    @readonlyWrapper
    def rmDataset(self,hdfdir,dataset):
        if hdfdir in self.fileObj:
            g = self.fileObj[hdfdir]
            if dataset in g.keys():
                del g[dataset]
        return

    def lsDatasets(self,hdfdir,recursive=False):
        ls = []
        thisdir = self.fileObj[hdfdir]
        for obj in thisdir.keys():            
            if isinstance(thisdir[obj],h5py.Group) and recursive:
                path = hdfdir+"/"+str(obj)
                path = path.replace("//","/")
                ls = ls + self.lsDatasets(path+"/",recursive=recursive)
            if isinstance(thisdir[obj],h5py.Dataset):
                if recursive:
                    path = hdfdir+"/"+str(obj)
                    path = path.replace("//","/")
                    ls.append(path)
                else:
                    ls.append(str(obj))
        return ls
    
    def findMatchingDatasets(self,hdfdir,searchItems,recursive=False,exit_if_missing=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if recursive:
            objs = self.lsGroup(hdfdir,recursive=recursive)                     
        else:
            objs = self.lsDatasets(hdfdir)           
        matches = findMatchingItems(objs,searchItems)        
        if exit_if_missing:
            missing = findMissingItems(matches,searchItems)
            #missing = list(set(matches).difference(objs))
            if len(missing) > 0:
                dashed = "-"*10
                err = dashed+"\nERROR! "+funcname+"(): No matches found for:"+\
                    hdfdir+":\n     "+"\n     ".join(missing)+"\n"+dashed
                print(err)
                raise KeyError(funcname+"(): Some required keys cannot be found in '"+hdfdir+"'!")
        return matches
    
    def datasetExists(self,hdfdir,name,exit_if_missing=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        exists = name in self.lsDatasets(hdfdir)
        if not exists and exit_if_missing:
            raise KeyError(funcname+"(): dataset '"+name+"' not found in "+hdfdir+"!")
        return exists

    def readDataset(self,hdfPath,exit_if_missing=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        name = hdfPath.split("/")[-1]
        hdfdir = hdfPath.replace(name,"")
        if hdfdir not in self.fileObj:
            raise KeyError(funcname+"(): "+hdfdir+" not found in HDF5 file!")        
        data = None
        if self.datasetExists(hdfdir,name,exit_if_missing=exit_if_missing):
            data = np.array(self.fileObj[hdfPath])
        return data

    def storeDataset(self,data,hdfdir,name,exit_if_missing=True):
        arr = self.readDataset(hdfdir+"/"+name,exit_if_missing=exit_if_missing)
        assert(arr.shape==data[name].shape)
        if arr is not None:
            data[name] = arr
        return

    def buildDataType(self,hdfdir,names):
        dtype = [(name,str(self.fileObj[hdfdir+"/"+name].dtype)) for name in names]
        return dtype

    def datasetSize(self,hdfPath,exit_if_missing=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        name = hdfPath.split("/")[-1]
        hdfdir = hdfPath.replace(name,"")
        size = 0
        if self.datasetExists(hdfdir,name,exit_if_missing=exit_if_missing):
            size = self.fileObj[hdfPath].size
        return size
            
    def readDatasets(self,hdfdir,recursive=False,required=None,exit_if_missing=True):
        """
        readDatasets(): Read one or more HDF5 datasets.

        USAGE:   data = HDF5().readDatasets(hdfdir,[recursive],[required],[exist_if_missing])
        
        Inputs:
               hdfdir : Path to dataset or group of datasets to read.
               recursive : If reading HDF5 group, read recursively down subgroups. 
                           (Default = False)
               required : List of names of datasets to read. If required=None, will read
                          all datasets. (Default = None).
               exit_if_missing : Will raise error and exit if any of names in 'required'
                                 are missing. (Default = True).
        
        Outputs:
               data : Numpy array of datasets.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if hdfdir not in self.fileObj:
            raise KeyError(funcname+"(): "+hdfdir+" not found in HDF5 file!")        
        if isinstance(self.fileObj[hdfdir],h5py.Dataset):            
            # Read single dataset
            DATA = self.readDataset(hdfdir,exit_if_missing=exit_if_missing)
        elif isinstance(self.fileObj[hdfdir],h5py.Group):
            # Read datasets in group
            # i) List datasets (recursively if specified)
            if recursive:
                objs = self.lsGroup(hdfdir,recursive=recursive)                     
            else:
                objs = self.lsDatasets(hdfdir)   
            if required is not None:                
                objs = self.findMatchingDatasets(hdfdir,required,recursive=recursive,\
                                                    exit_if_missing=exit_if_missing)
            # ii) Get datatypes
            dtype = self.buildDataType(hdfdir,objs)
            # iii) Initialize array
            n = self.datasetSize(hdfdir+"/"+objs[0])
            DATA = np.zeros(n,dtype=dtype)
            # ii) Store datasets in array
            dummy = [self.storeDataset(DATA,hdfdir,obj) for obj in objs]
        return DATA
                            
    
    ##############################################################################
    # ATTRIBUTES
    ##############################################################################
    
    def readAttributes(self,hdfdir,required=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if required is None:        
            attr = dict(self.fileObj[hdfdir].attrs)
        else:
            good = list(set(required).intersection(self.fileObj[hdfdir].attrs.keys()))
            bad = list(set(required).difference(self.fileObj[hdfdir].attrs.keys()))            
            if self.verbose:
                if len(bad)>0:
                    warnings.warn(funcname+"(): Following attributes not present in '"+
                                  hdfdir+"':"+",".join(bad))
            if len(good)==0:
                return {}
            attr = {g:self.fileObj[hdfdir].attrs[g] for g in good}
        if six.PY3:
            attr = {removeByteStrings(key):removeByteStrings(value) for key,value in attr.items()}
        return attr
    
    @readonlyWrapper
    def addAttributes(self,hdfdir,attributes,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if hdfdir not in self.fileObj:
            raise KeyError(funcname+"(): '"+hdfdir+"' not found in HDF5 file!")                
        attrib = self.fileObj[hdfdir].attrs        
        for att in attributes.keys():
            if att in self.fileObj[hdfdir].attrs.keys():
                if self.verbose:
                        warnings.warn(funcname+"(): Attribute '"+att+"' already exists!")
                if overwrite:
                    if self.verbose:
                        print("        Overwriting attribute '"+att+"'")                        
                    if six.PY3:
                        attrib.create(att,addByteStrings(attributes[att]),shape=None,dtype=None)
                    else:
                        attrib.create(att,attributes[att],shape=None,dtype=None)
                else:
                    if self.verbose:
                        print("        Ignoring attribute '"+att+"'")                        
            else:
                if six.PY3:
                    attrib.create(att,addByteStrings(attributes[att]),shape=None,dtype=None)
                else:
                    attrib.create(att,attributes[att],shape=None,dtype=None)
        return

    @readonlyWrapper
    def rmAttributes(self,hdfdir,attributes=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if hdfdir not in self.fileObj:
            raise KeyError(funcname+"(): '"+hdfdir+"' not found in HDF5 file!")                
        attrib = self.fileObj[hdfdir].attrs
        if attributes is None:
            attributes = attrib.keys()
        [attrib.__delitem__(att) for att in attributes if att in attrib.keys()]
        return

