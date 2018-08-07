#! /usr/bin/env python

import os,fnmatch
import warnings
from . import rcParams

def locateDatasetsRepository():
    # Load path to Galacticus datasets
    DATASETS_PATH = None
    key = "GALACTICUS_DATA_PATH"
    DATASETS_PATH = rcParams.get("paths",key,fallback=None)
    if DATASETS_PATH == "None":
        msg = "No path specified for Galacticus datasets. "+\
            "Specify the path in your environment variables "+\
            "using the variable name '"+datasetsKeyName+"'."
        raise RuntimeError(msg)
    if not DATASETS_PATH.endswith("/"):
        DATASETS_PATH = DATASETS_PATH + "/"
    return DATASETS_PATH


def recursiveGlob(treeroot,pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results


class GalacticusData(object):

    def __init__(self,verbose=True):
        self.verbose = verbose
        self.path = locateDatasetsRepository()
        # Check that the path exists
        if not os.path.exists(self.path):
            msg = "Datasets path '"+self.path+"' does not exist."
            raise RuntimeError(msg)    
        # Check path ends with a forward-slash
        if not self.path.endswith("/"):
            self.path = self.path + "/"
        # Check that the static subdirectory exists
        self.static = self.path + "static/"
        if not os.path.exists(self.static):
            msg = "ERROR! Static datasets path '"+DATASETS_PATH+"static' does not exist."
            raise RuntimeError(msg)
        # Make dynamic path if not found
        self.dynamic = self.path + "dynamic/"
        if not os.path.exists(self.dynamic):
            os.makedirs(self.dynamic)
        return

    def __call__(self,pattern):
        return self.search(pattern)

    def _searchDirectory(self,path,pattern,errorNotFound=True):
        files = recursiveGlob(path,pattern)
        if len(files)>1:            
            msg = "Multilple files found matching pattern "+pattern+"."
            if self.verbose:
                msg = msg + "DIRECTORY = "+path+"\nFiles found are:\n"+\
                    "\n".join(files)
            warnings.warn(msg)        
        if len(files)==0:
            if errorNotFound:
                msg = "No files found in "+self.static+\
                    " matching pattern "+pattern+"."
                raise RuntimeError(msg)
            return None
        if self.verbose:
            print("Returning file: "+files[0])
        return files[0]

    def searchStatic(self,pattern,errorNotFound=True):
        return self._searchDirectory(self.static,pattern,errorNotFound=errorNotFound)

    def searchDynamic(self,pattern,errorNotFound=False):
        return self._searchDirectory(self.dynamic,pattern,errorNotFound=errorNotFound)
    
    def search(self,pattern):
        dataFile = self.searchDynamic(pattern,errorNotFound=False)
        if dataFile is not None:
            return dataFile
        dataFile = self.searchStatic(pattern,errorNotFound=True)
        return dataFile
    
    


    
    


