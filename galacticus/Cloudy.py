#! /usr/bin/env python

import sys
import numpy as np
from scipy.interpolate import interpn
from .data import GalacticusData
from .fileFormats.hdf5 import HDF5


class CloudyTable(HDF5):
    """
    CloudyTable: Class to read and interpolate over a table of luminosities output from CLOUDY. The 
                 class assumes that the CLOUDY table is stored in a file with name 'emissionLines.hdf5'.

    USAGE: CLOUDY = CloudyTable([path=None],[verbose=False])

         INPUT 
                path -- Path to datasets repository. If None, will search for path in environment 
                        variables (stored as 'GALACTICUS_DATASETS'). [Default=None] 
             verbose -- Print additional information. [Default=False]

             
         Functions:
                   getInterpolant(): Extract values for specified interpolant.
                   getWavelength(): Get rest wavelength of specified emission line.
                   reportLimits(): Report limits of interpolants.
                   interpolate(): Interpolate the table of CLOUDY outputs for specified 
                                  emission line.
    """
    def __init__(self,path=None,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set verbosity
        self.verbose = verbose
        # Locate table file
        DATA = GalacticusData(verbose=self.verbose)
        cloudyFile = DATA.search("emissionLines.hdf5")        
        # Initalise HDF5 class and open emissionLines.hdf5 file
        super(cloudyTable, self).__init__(cloudyFile,'r')
        # Extract names and properties of lines
        self.lines = list(map(str,self.fileObj["lines"].keys()))
        self.wavelengths = {}
        self.luminosities = {}
        for l in self.lines:
            self.wavelengths[l] = self.readAttributes("lines/"+l,required=["wavelength"])["wavelength"]
            self.luminosities[l] = self.readDatasets('lines',required=[l])[l]        
        # Store interpolants
        self.interpolants = ["metallicity","densityHydrogen","ionizingFluxHydrogen",\
                                 "ionizingFluxHeliumToHydrogen","ionizingFluxOxygenToHydrogen"]
        self.interpolantsData = ((np.log10(self.readDatasets('/',required=[name])[name]),) \
                                     for name in self.interpolants)            
        return


    def _getInterpolantRange(self,interpolantName):
        values = self.getInterpolant(interpolantName)
        return values.min(),values.max()


    def getInterpolant(self,interpolant):
        """
        getInterpolant(): Return data for specified interpolant.

        USAGE: data = CloudyTable().getInterpolant(interpolant)

            INPUT
                 interpolant -- Name of interpolant. List of available interpolants
                                can be found by viewing the 'interpolants' class 
                                attribute.
        
            OUTPUT
                 data        -- Numpy array of interpolant data.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if interpolant not in self.interpolants:
            raise ValueError(funcname+"(): interpolant '"+interpolant+"'not recognised! Options are: "+\
                                 ",".join(self.interpolants))
        return np.log10(self.readDatasets('/',required=[interpolant])[interpolant])        


    def getWavelength(self,lineName):
        """
        getWavelength(): Return rest wavelength for specified emission line. Note that the rest
                         wavelength stored by CLOUDY may differ slihgtly from the true rest
                         wavelength -- this may be due to rounding errors or a bug in CLOUDY.

        USAGE: data = CloudyTable().getWavelength(line)

            INPUT
                 line -- Name of emission line. List of available lines
                         can be found by viewing the 'lines' class attribute.
        
            OUTPUT
                 data -- Float value for rest wavelength.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        return self.wavelengths[lineName]


    def reportLimits(self,data=None):
        """
        reportLimits(): Print report of ranges for interpolants in CLOUDY table.

        USAGE: CloudyTable().reportLimits([data=None])

             INPUT
                  data -- Numpy array containing galaxy data, with shape (n,m)
                          where n is number of interpolants and m is number
                          of galaxies. Array can be constructed using 
                          CloudyTable().prepareGalaxyData().

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        ndash = 40
        print("-"*ndash)
        print("CLOUDY Interpolation Report:")
        for i,name in enumerate(self.interpolants):
            print("("+str(i)+1+") "+name)        
            print("CLOUDY Range (min,max) = "+str(self.interpolantsData[i].min())+", "+\
                      str(self.interpolantsData[i].max()))
            if data is not None:        
                if name in data.dtype.names:
                    print("Galaxy Data (min,max,median) = "+str(data[i,:].min())+", "+str(data[i,:].max())\
                              +", "+str(np.median(data[i,:])))
        print("-"*ndash)
        return


    def prepareGalaxyData(self,metallicity,densityHydrogen,ionizingFluxHydrogen,\
                        ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen):
        """
        prepareGalaxyData(): Function to zip galaxy data ready for input into scipy.interpolate.interpn 
                             (used to interpolate CLOUDY table).

        USAGE:  data = CloudyTable().prepareGalaxyData(metallicity,densityHydrogen,ionizingFluxHydrogen,\
                        ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen)
                                                     
             INPUT
                   metallicity                  -- Numpy array of galaxy metallicites.
                   densityHydrogen              -- Numpy array of galaxy hydrogen gas density.
                   ionizingFluxHydrogen         -- Numpy array of galaxy hydrogen ionizing flux.
                   ionizingFluxHeliumToHydrogen -- Numpy array of galaxy helium ionizing flux.
                   ionizingFluxOxygenToHydrogen -- Numpy array of galaxy oxygen ionizing flux.

             OUTPUT
                   data                         -- Numpy array of zipped galaxy data.
                             
        """
        return zip(metallicity,densityHydrogen,ionizingFluxHydrogen,\
                       ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen)
    
    
    def interpolate(self,lineName,metallicity,densityHydrogen,ionizingFluxHydrogen,\
                        ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen,**kwargs):
        """
        interpolate(): Interpolate a table of CLOUDY output for specified emission line.

        USAGE:  luminosity = CloudyTable().interpolate(line,metallicity,densityHydrogen,ionizingFluxHydrogen,\
                                                       ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen,\
                                                       **kwargs)
                                                       
               INPUT
                    line                         -- Name of emission line. Available lines provided in 'lines'
                                                    class attribute.
                    metallicity                  -- Numpy array of galaxy metallicites.
                    densityHydrogen              -- Numpy array of galaxy hydrogen gas density.
                    ionizingFluxHydrogen         -- Numpy array of galaxy hydrogen ionizing flux.
                    ionizingFluxHeliumToHydrogen -- Numpy array of galaxy helium ionizing flux.
                    ionizingFluxOxygenToHydrogen -- Numpy array of galaxy oxygen ionizing flux.
                    **kwargs                     -- Keyword arguments to pass to scipy.interpolate.interpn.

              OUTPUT
                    luminosity                   -- Numpy array of emission line luminosities.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        tableLuminosities = self.luminosities[lineName]
        galaxyData = self.prepareGalaxyData(metallicity,densityHydrogen,ionizingFluxHydrogen,\
                                                ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen)
        if "bounds_error" not in kwargs.keys():
            kwargs["bounds_error"] = False
        if "fill_value" not in kwargs.keys():
            kwargs["fill_value"] = None
        if self.verbose:
            self.reportLimits(data=galaxyData)
        luminosities = interpn(self.interpolantsData,tableLuminosities,galaxyData,**kwargs)
        return luminosities
    

