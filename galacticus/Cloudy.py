#! /usr/bin/env python

"""
galacticus.Cloudy
=================

Classes for working with library of Cloudy output.
"""

import sys
import fnmatch
import numpy as np
import unittest
import warnings
import six
from scipy.interpolate import interpn
from .data import GalacticusData
from .fileFormats.hdf5 import HDF5
from . import rcParams


class CloudyEmissionLine(object):
    """
    Class to store emission line data from a Cloudy HDF5 library.
                          
    Arguments:
        name (str) : Name of emission line
        wavelength (float) : Wavelength of emission line as computed by Cloudy.
        luminosities (array_like, {N,N,N,N,N}) : Five dimensional array storing luminosities for this emission line as function of various interpolants.

    """
    def __init__(self,name=None,wavelength=None,luminosities=None):
        self.name = name
        self.wavelength = wavelength
        self.luminosities = luminosities
        return

    def reset(self):
        """
        Reset the :class:`~galaticus.Cloudy.CloudyEmissionLine` instance.
        """
        self.name = None
        self.wavelength = None
        self.luminosities = None


class CloudyTable(HDF5):
    """
    Read and interpolate over a library of luminosities output from Cloudy. The class assumes that
    by default the Cloudy table **emissionLines.hdf5**. The name of the file can be modified using
    **rcParams**, an instance of :class:`~galacticus.rcConfig`.

    Arguments:
        verbose (bool,optional) : Print additional information.

    Attributes:
        lines (dictionary) : Dictionary storing Cloudy emission line data.
        interpolants (list,str) : List of interpolants.
        interpolantsData (tuple,array_like) : Tuple of arrays of interpolant values. 

    """
    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set verbosity
        self.verbose = verbose
        # Locate table file
        DATA = GalacticusData(verbose=self.verbose)
        fileName = rcParams.get("cloudy","fileName",fallback="emissionLines.hdf5")
        cloudyFile = DATA.search(fileName)        
        # Initalise HDF5 class and open emissionLines.hdf5 file
        super(CloudyTable, self).__init__(cloudyFile,'r')
        # Extract names and properties of lines
        self.lines = {}
        # Store interpolants
        self.interpolants = ["metallicity","densityHydrogen","ionizingFluxHydrogen",\
                                 "ionizingFluxHeliumToHydrogen","ionizingFluxOxygenToHydrogen"]
        self.interpolantsData = None
        return

    def listAvailableLines(self):
        """
        List emission lines available in Cloudy HDF5 library file.

        Returns:
            list,str : List of emission line names.
        """
        return self.lsDatasets("/lines")
    
    def loadEmissionLine(self,line):
        """
        Load data for specified emission line from Cloudy library file into an instance of :class:`~galacticus.Cloudy.CloudyEmissionLine`.
        Store :class:`~galacticus.Cloudy.CloudyEmissionLine` instance in :attr:`~lines` attribute.

        Arguments:
            line (str) : Name of specified emission line.
                                        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if line not in self.lsDatasets("/lines"):
            warnings.warn(funcname+"(): Emission line '"+line+"' not found in Cloudy output.")
            return 
        LINE = CloudyEmissionLine(name=line)
        LINE.wavelength = self.readAttributes("lines/"+line,required=["wavelength"])["wavelength"]
        LINE.luminosities = self.readDataset('/lines/'+line)
        self.lines[line] = LINE
        return
    
    def loadEmissionLines(self):
        """
        Load all emission lines data from Cloudy HDF5 file. Function :meth:`~loadEmissionLine` is 
        used to store emission line data in :attr:`~lines` attribute.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        [self.loadEmissionLine(l) for l in self.lsDatasets("/lines") if l not in self.lines.keys()]
        return

    def getInterpolant(self,interpolant):
        """
        Load log10 of data for specified interpolant from Cloudy HDF5 file.

        Arguments:
            interpolant (str) : Name of interpolant.
            
        Returns:
            array_like,{N,} : Numpy array of interpolant data.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if interpolant not in self.interpolants:
            msg = funcname+"(): interpolant '"+interpolant+"'not recognised!"
            msg = msg + " Options are: "+",".join(self.interpolants)
            raise KeyError(msg)
        return np.log10(self.readDataset('/'+interpolant))

    def loadInterpolantsData(self):
        """
        Load all interpolants data from Cloudy HDF5 file. Function :meth:`~getInterpolant` is
        used to store a tuple of Numpy arrays, sotring the data for each interpolant.        
        """
        self.interpolantsData = tuple([self.getInterpolant(name) for name in self.interpolants])            
        return

    def getWavelength(self,lineName):
        """
        Return the wavelength, in Angstroms, for the specified emission line.

        Arguments:
             lineName (str) : Name of emission line.
             
        Return:
             float : Wavelength in Angstroms.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lsDatasets("/lines"):
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        if lineName not in self.lines.keys():
            self.loadEmissionLine(lineName)
        return self.lines[lineName].wavelength


    def reportLimits(self,data=None):
        """
        Report limits of Cloudy interpolants and optional range of galaxy data.

        Arguments:
            data (list,optional) : List of galaxy data constucted using :meth:`~prepareGalaxyData`.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.interpolantsData is None:
            self.loadInterpolantsData()
        ndash = 40
        print("-"*ndash)
        print("CLOUDY Interpolation Report:")
        for i,name in enumerate(self.interpolants):
            print("("+str(i+1)+") "+name)        
            print("  CLOUDY Range (min,max) = "+str(self.interpolantsData[i].min())+", "+\
                      str(self.interpolantsData[i].max()))
            if data is not None:        
                if name in data.dtype.names:
                    print("  Galaxy Data (min,max,median) = "+str(data[i,:].min())+", "+\
                              str(data[i,:].max())+", "+str(np.median(data[i,:])))
        print("-"*ndash)
        return


    def prepareGalaxyData(self,metallicity,densityHydrogen,ionizingFluxHydrogen,\
                        ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen):
        """
        Return zipped list of galaxy data.

        Arguments:
            metallicity (array_like,{N,}) : Numpy array of galaxy metallicity.
            densityHydrogen (array_like,{N,}) : Numpy array of galaxy hydrogen gas density.
            ionizingFluxHydrogen (array_like,{N,}) : Numpy array of galaxy Lyman ionizing luminosity.            
            ionizingFluxHeliumToHydrogen (array_like,{N,}) : Numpy array of ratio for galaxy He/Lyman ionizing luminosities.
            ionizingFluxOxygenToHydrogen (array_like,{N,}) : Numpy array of ratio for galaxy O/Lyman ionizing luminosities.
            
        Return:
            list : List of Numpy arrays for galaxy properties.

        """
        Z = zip(metallicity,densityHydrogen,ionizingFluxHydrogen,
                ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen)
        if not six.PY2:
            Z = list(Z)
        return Z
    
    def interpolate(self,lineName,metallicity,densityHydrogen,ionizingFluxHydrogen,\
                        ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen):
        """
        Interpolate over library of Cloudy HDF5 file to obtain luminosity for specified emission line.

        Arguments:
            lineName (str) : Emission line name.
            metallicity (array_like,{N,}) : Numpy array of galaxy metallicity.
            densityHydrogen (array_like,{N,}) : Numpy array of galaxy hydrogen gas density.
            ionizingFluxHydrogen (array_like,{N,}) : Numpy array of galaxy Lyman ionizing luminosity.            
            ionizingFluxHeliumToHydrogen (array_like,{N,}) : Numpy array of ratio for galaxy He/Lyman ionizing luminosities.
            ionizingFluxOxygenToHydrogen (array_like,{N,}) : Numpy array of ratio for galaxy O/Lyman ionizing luminosities.

        Return:
            array_like,{N,} : NUmpy array of galaxy luminosity for specified emission line.

        Note:
            The keyword arguments **bounds_error**, **fill_value** and **method** for
            `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
            can be set using **rcParams**. For example
        
            >>> from galacticus import rcParams
            >>> rcParams.set("cloudy","bounds_error",False)
            >>> rcParams.set("cloudy","fill_value",None)
            >>> rcParams.set("cloudy","method",'linear')

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lines.keys():
            self.loadEmissionLine(lineName)
        tableLuminosities = self.lines[lineName].luminosities
        galaxyData = self.prepareGalaxyData(metallicity,densityHydrogen,ionizingFluxHydrogen,\
                                                ionizingFluxHeliumToHydrogen,\
                                                ionizingFluxOxygenToHydrogen)
        bounds_error = rcParams.getboolean("cloudy","bounds_error",fallback=False)
        method = rcParams.get("cloudy","method",fallback='linear')
        fill_value = rcParams.get("cloudy","fill_value",fallback=None)
        fill_value = str(fill_value)
        if fnmatch.fnmatch(fill_value.lower(),"none"):
            fill_value = None
        elif fnmatch.fnmatch(fill_value.lower(),"nan"):            
            fill_value = np.nan
        else:
            fill_value = float(fill_value)
        if self.verbose:
            self.reportLimits(data=galaxyData)
        if self.interpolantsData is None:
            self.loadInterpolantsData()
        luminosities = interpn(self.interpolantsData,tableLuminosities,galaxyData,\
                                   method=method,bounds_error=bounds_error,\
                                   fill_value=fill_value)
        return luminosities
    
