#! /usr/bin/env python

"""
galacticus.data
===============

Module to locate data files from the `Galacticus datasets <https://bitbucket.org/galacticusdev/datasets/>`_ repository.
To use the datasets you will need to clone the repository using:

>>> hg clone https://aimerson@bitbucket.org/galacticusdev/datasets

Once the repository is downloaded, you will need to set the environment variable `GALACTICUS_DATA_PATH`
to point to the location of the repository, e.g.

For csh/tcsh:

>>> setenv GALACTICUS_DATA_PATH /path/to/datasets

For bash:

>>> export GALACTICUS_DATA_PATH="/path/to/datasets"

"""
import unittest
import sys,os,fnmatch
import warnings
from . import rcParams


def recursiveGlob(treeroot,pattern):
    """
    Perform a recursive `glob` search.
    
    Arguments:
        treeroot (str) : Root directory within which to search.
        pattern (str) : Pattern to search for files.

    Return:
        list,str : List containing paths to files matching pattern search.
        
    """
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results


class GalacticusData(object):
    
    """
    Base class to locate files in the Galacticus datasets repository.
    
    Arguments:
        verbose (bool,optional) : Print additional information. 

    Attributes:
        verbose (bool) : Print additional information.
        path (str) : Path to datasets repository.
        static (str) : Path to the `static` datasets repository.
        dynamic (str) : Path to the `dynamic` datasets repository.

    """

    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        self.path = self.locateDatasetsRepository()
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
            msg = funcname+"(): Static datasets path '"+self.path+"/static' does not exist."
            raise RuntimeError(msg)
        # Make dynamic path if not found
        DYNAMIC_PATH = rcParams.get("path","GALACTICUS_DYNAMIC_DATA_PATH",fallback=None)
        if DYNAMIC_PATH is not None:
            self.dynamic = DYNAMIC_PATH
        else:
            self.dynamic = self.path + "dynamic/"
        if not os.path.exists(self.dynamic):
            os.makedirs(self.dynamic)
        return

    def __call__(self,pattern):
        """
        Calls :meth:`~search` method.
        """
        return self.search(pattern)

    @classmethod
    def locateDatasetsRepository(cls):
        """
        Locate datasets repository.
        
        Return:
            str : Path to datasets repository.
        
        Raises:
            RuntimeError : Raised if `GALACTICUS_DATA_PATH` is not set.
        
        """
        # Load path to Galacticus datasets
        DATASETS_PATH = None
        key = "GALACTICUS_DATA_PATH"
        DATASETS_PATH = rcParams.get("paths",key,fallback=None)
        if str(DATASETS_PATH) == "None":
            msg = "No path specified for Galacticus datasets. "+\
                "Specify the path in your environment variables "+\
                "using the variable name '"+key+"'."
            raise RuntimeError(msg)
        if not DATASETS_PATH.endswith("/"):
            DATASETS_PATH = DATASETS_PATH + "/"
        return DATASETS_PATH



    def _searchDirectory(self,path,pattern,errorNotFound=True):
        """
        Apply a recursive glob search within a specified directory for files matching
        the specified pattern.

        Arguments:
            path (str) : Parent directory to search within.
            pattern (str) : Pattern to search for files.
            errorNotFound (bool,optional) : Raises an error if no files are found.

        Raises:
            RuntimeError : Optionally raised if no files are found.

        """
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        files = recursiveGlob(path,pattern)
        if len(files)>1:            
            msg = classname+"(): Multiple files found matching pattern "+pattern+"."
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
            print(classname+"(): Returning file: "+files[0])
        return files[0]

    def searchStatic(self,pattern,errorNotFound=True):
        """
        Searches `datasets/static` directory for files matching specified pattern.

        Arguments:
            pattern (str): Pattern to search for files.
            errorNotFound (bool,optional) : Raises error if no matching files are found.

        Return:
            list,str : List containing paths to files matching pattern search.       

        Raises:
            RuntimeError : Optionally raised if no files are found.

        """
        return self._searchDirectory(self.static,pattern,errorNotFound=errorNotFound)

    def searchDynamic(self,pattern,errorNotFound=False):
        """
        Searches `datasets/dynamic` directory for files matching specified pattern.

        Arguments:
            pattern (str): Pattern to search for files.
            errorNotFound (bool,optional) : Raises error if no matching files are found.

        Return:
            list,str : List containing paths to files matching pattern search.       

        Raises:
            RuntimeError : Optionally raised if no files are found.


        """
        return self._searchDirectory(self.dynamic,pattern,errorNotFound=errorNotFound)
    
    def search(self,pattern,dynamicError=False,staticError=True):
        """
        Searches in `datasets/dynamic` and `datasets/static` directories for files matching specified pattern. 
        The dynamic directory is searched first. The static directory is searched only if no files are found
        in the dynamic directory.

        Arguments:
            pattern (str): Pattern to search for files.
            dynamicError (bool,optional) : Raises error if no matching files are found in dynamic directory.
            staticError (bool,optional) : Raises error if no matching files are found in static directory.

        Return:
            list,str : List containing paths to files matching pattern search.       

        Raises:
            RuntimeError : Optionally raised if no files are found.

        """
        dataFile = self.searchDynamic(pattern,errorNotFound=dynamicError)
        if dataFile is not None:
            return dataFile
        dataFile = self.searchStatic(pattern,errorNotFound=staticError)
        return dataFile
    

