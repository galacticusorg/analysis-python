#! /usr/bin/env python

import sys
import fnmatch
import numpy as np
import unittest
import warnings
from scipy.interpolate import interpn
from .data import GalacticusData
from .fileFormats.hdf5 import HDF5
from . import rcParams


class CloudyEmissionLine(object):
    """
    CloudyEmissionLine(): Class to store emission line data as read from 
                          the Cloudy HDF5 file.
                          
    Attributes:
            name         -- Name of emission line
            wavelength   -- Wavelength of emission line as computed by Cloudy.
            luminosities -- 5-dimensional array storing luminosities for this 
                            emission line as function of various interpolants.

    """
    def __init__(self,name=None,wavelength=None,luminosities=None):
        self.name = name
        self.wavelength = wavelength
        self.luminosities = luminosities
        return

    def reset(self):
        self.name = None
        self.wavelength = None
        self.luminosities = None


class CloudyTable(HDF5):
    """
    CloudyTable: Class to read and interpolate over a table of luminosities output from Cloudy. The 
                 class assumes that by default the Cloudy table is stored in a file with name 
                 'emissionLines.hdf5'. The name of the file can be modified using the rcParams.

    USAGE: CLOUDY = CloudyTable([verbose=False])

         INPUT 
             verbose -- Print additional information. [Default=False]

         Functions:
                   loadEmissionLine(): Load specified emission line from HDF5 file.
                   loadEmissionLines(): Load all emission lines found in HDF5 file.                   
                   getInterpolant(): Extract values for specified interpolant.
                   loadInterpolantsData(): Load all interpolants data stored in HDF5 file.
                   getWavelength(): Get rest wavelength of specified emission line.
                   reportLimits(): Report limits of interpolants.
                   prepareGalaxyData(): Zip galaxy data ready for interpolation.
                   interpolate(): Interpolate the table of Cloudy outputs for specified 
                                  emission line.
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

    def loadEmissionLine(self,line):
        """
        CloudyTable.loadEmissionLine(): Reads the emission line data from the Cloudy HDF5 file
                                        and stores in the CloudyTable.lines dictionary. The 
                                        information is stored as an instance of the 
                                        'CloudyEmissionLine' class.

        USAGE: CloudyTable.loadEmissionLine(line)

            INPUT
                line -- Name of emission line to read. If line cannot be found no new
                        information will be added to CloudyTable.lines.
                                        
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
        CloudyTable.loadEmissionLines(): Read all emission lines from Cloudy HDF5 file. All data
                                         is stored in CloudyTable.lines dictionary as instances
                                         of the 'CloudyEmissionLine' class.

        USAGE: CloudyTable.loadEmissionLines()

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        [self.loadEmissionLine(l) for l in self.lsDatasets("/lines") if l not in self.lines.keys()]
        return

    def getInterpolant(self,interpolant):
        """
        CloudyTable.getInterpolant(): Return data for specified interpolant.

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
            msg = funcname+"(): interpolant '"+interpolant+"'not recognised!"
            msg = msg + " Options are: "+",".join(self.interpolants)
            raise KeyError(msg)
        return np.log10(self.readDataset('/'+interpolant))

    def loadInterpolantsData(self):
        """
        CloudyTable.loadInterpolantsData(): Load bin values for each of the interpolants into
                                            CloudyTable.interpolantsData where data is stored
                                            as tuple of Numpy arrays for input into 
                                            CloudyTable.interpolate. To extract data as 
                                            individual Numpy arrays use 
                                            CloudyTable.getInterpolant.
        
        USAGE: CloudyTable.loadInterpolantsData()
        
        """
        self.interpolantsData = tuple([self.getInterpolant(name) for name in self.interpolants])            
        return

    def getWavelength(self,lineName):
        """
        CloudyTable.getWavelength(): Return rest wavelength for specified emission line. Note that 
                                     the rest wavelength stored by CLOUDY may differ slihgtly from 
                                     the true rest wavelength -- this may be due to rounding 
                                     errors or a bug in CLOUDY.

        USAGE: data = CloudyTable().getWavelength(line)

            INPUT
                 line -- Name of emission line. List of available lines
                         can be found by viewing the 'lines' class attribute.
        
            OUTPUT
                 data -- Float value for rest wavelength.
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lsDatasets("/lines"):
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        if lineName not in self.lines.keys():
            self.loadEmissionLine(lineName)
        return self.lines[lineName].wavelength


    def reportLimits(self,data=None):
        """
        CloudyTable.reportLimits(): Print report of ranges for interpolants in CLOUDY table.

        USAGE: CloudyTable().reportLimits([data=None])

             INPUT
                  data -- Numpy array containing galaxy data, with shape (n,m)
                          where n is number of interpolants and m is number
                          of galaxies. Array can be constructed using 
                          CloudyTable().prepareGalaxyData().

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
        CloudyTable.prepareGalaxyData(): Function to zip galaxy data ready for input into 
                                         scipy.interpolate.interpn (used to interpolate over 
                                         CLOUDY table).

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
                        ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen):
        """
        CloudyTable.interpolate(): Interpolate a table of CLOUDY output for specified emission line.

        USAGE:  luminosity = CloudyTable().interpolate(line,metallicity,densityHydrogen,ionizingFluxHydrogen,\
                                                       ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen)
                                                       
               INPUT
                    line                         -- Name of emission line. Available lines provided in 'lines'
                                                    class attribute.
                    metallicity                  -- Numpy array of galaxy metallicites.
                    densityHydrogen              -- Numpy array of galaxy hydrogen gas density.
                    ionizingFluxHydrogen         -- Numpy array of galaxy hydrogen ionizing flux.
                    ionizingFluxHeliumToHydrogen -- Numpy array of galaxy helium ionizing flux.
                    ionizingFluxOxygenToHydrogen -- Numpy array of galaxy oxygen ionizing flux.

              OUTPUT
                    luminosity                   -- Numpy array of emission line luminosities.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")        
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
    

class UnitTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.CLOUDY = CloudyTable()
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory
        self.CLOUDY.close()
        del self.CLOUDY
        return

    def testLoadEmissionLine(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Cloudy: "+funcname)
        print("Testing Cloudy.loadEmissionLine() function")
        name = "balmerAlpha6563"
        self.CLOUDY.loadEmissionLine(name)
        self.assertTrue(name in self.CLOUDY.lines.keys())
        LINE = self.CLOUDY.lines[name]
        self.assertEqual(LINE.name,name)
        wavelength = self.CLOUDY.readAttributes("lines/"+name,required=["wavelength"])["wavelength"]
        self.assertEqual(wavelength,LINE.wavelength)
        luminosities = self.CLOUDY.readDataset('/lines/'+name)        
        self.assertEqual(luminosities.shape,LINE.luminosities.shape)
        diff = np.fabs(luminosities-LINE.luminosities).flatten()
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            badName = "notAnEmissionLine9999"
            self.CLOUDY.loadEmissionLine(badName)
            self.assertFalse(badName in self.CLOUDY.lines.keys())                
        print("TEST COMPLETE")
        print("\n")                
        return

    def testLoadEmissionLines(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Cloudy: "+funcname)
        print("Testing Cloudy.loadEmissionLines() function")
        self.CLOUDY.lines = {}
        self.CLOUDY.loadEmissionLines()
        lines = self.CLOUDY.lsDatasets("/lines")
        self.assertEqual(len(lines),len(self.CLOUDY.lines.keys()))
        [self.assertTrue(line in self.CLOUDY.lines.keys()) for line in lines]
        [self.assertIsNotNone(self.CLOUDY.lines[l]) for l in lines]
        print("TEST COMPLETE")
        print("\n")
        return
    
    def testGetWavelength(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Cloudy: "+funcname)
        print("Testing Cloudy.getWavelength() function")
        self.CLOUDY.loadEmissionLines()
        for name in self.CLOUDY.lsDatasets("/lines"):
            value = self.CLOUDY.lines[name].wavelength
            wavelength = self.CLOUDY.getWavelength(name)
            self.assertEqual(value,wavelength)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            badName = "notAnEmissionLine9999"
            self.assertRaises(IndexError,self.CLOUDY.getWavelength,badName)
        print("TEST COMPLETE")
        print("\n")
        return

    
    def testGetInterpolant(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Cloudy: "+funcname)
        print("Testing Cloudy.getInterpolant() function")
        for name in self.CLOUDY.interpolants:
            data = self.CLOUDY.getInterpolant(name)
            values = np.log10(self.CLOUDY.readDataset('/'+name))
            diff = data - values
            [self.assertLessEqual(d,1.0e-6) for d in diff]
        self.assertRaises(KeyError,self.CLOUDY.getInterpolant,"someGas")
        print("TEST COMPLETE")
        print("\n")
        return


    def testLoadInterpolantsData(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Cloudy: "+funcname)
        print("Testing Cloudy.loadInterpolantsData() function")
        self.CLOUDY.interpolantsData = None
        self.CLOUDY.loadInterpolantsData()
        self.assertIsNotNone(self.CLOUDY.interpolantsData)
        print("TEST COMPLETE")
        print("\n")
        return

    def testPrepareGalaxyData(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Cloudy: "+funcname)
        print("Testing Cloudy.prepareGalaxyData() function")
        N = 1000
        values = np.random.rand(N)
        data = self.CLOUDY.prepareGalaxyData(values,values,values,values,values)
        self.assertEqual(len(data),N)
        self.assertEqual(len(data[0]),5)
        self.assertRaises(TypeError,self.CLOUDY.prepareGalaxyData,values)
        print("TEST COMPLETE")
        print("\n")
        return

    def testInterpolate(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Cloudy: "+funcname)
        print("Testing Cloudy.Interpolate() function")
        # Test incorrect calling of function
        badName= "notAnEmissionLine999"
        N = 1000
        metallicity = np.random.rand(N)
        densityHydrogen = np.random.rand(N)
        ionizingFluxHydrogen = np.random.rand(N)
        ionizingFluxHeliumToHydrogen = np.random.rand(N)
        ionizingFluxOxygenToHydrogen = np.random.rand(N)        
        self.assertRaises(IndexError,self.CLOUDY.interpolate,badName,metallicity,\
                              densityHydrogen,ionizingFluxHydrogen,\
                              ionizingFluxHeliumToHydrogen,\
                              ionizingFluxOxygenToHydrogen)    
        name = "balmerAlpha6563"
        self.assertRaises(TypeError,name,metallicity)
        # Generate random data to test correct calling
        i = self.CLOUDY.getInterpolant("metallicity")
        diff = i.max() - i.min()
        metallicity = np.random.rand(N)*diff + i.min()
        i = self.CLOUDY.getInterpolant("densityHydrogen")
        diff = i.max() - i.min()
        densityHydrogen = np.random.rand(N)*diff + i.min()
        i = self.CLOUDY.getInterpolant("ionizingFluxHydrogen")
        diff = i.max() - i.min()
        ionizingFluxHydrogen = np.random.rand(N)*diff + i.min()
        i = self.CLOUDY.getInterpolant("ionizingFluxHeliumToHydrogen")
        diff = i.max() - i.min()
        ionizingFluxHeliumToHydrogen = np.random.rand(N)*diff + i.min()
        i = self.CLOUDY.getInterpolant("ionizingFluxOxygenToHydrogen")
        diff = i.max() - i.min()
        ionizingFluxOxygenToHydrogen = np.random.rand(N)*diff + i.min()
        # Test interpolation
        luminosity = self.CLOUDY.interpolate(name,metallicity,\
                                                 densityHydrogen,\
                                                 ionizingFluxHydrogen,\
                                                 ionizingFluxHeliumToHydrogen,\
                                                 ionizingFluxOxygenToHydrogen)
        self.assertIsInstance(luminosity,np.ndarray)
        metallicity[0] *= 1000.0
        rcParams.update("cloudy","fill_value","nan")
        luminosity = self.CLOUDY.interpolate(name,metallicity,\
                                                 densityHydrogen,\
                                                 ionizingFluxHydrogen,\
                                                 ionizingFluxHeliumToHydrogen,\
                                                 ionizingFluxOxygenToHydrogen)
        self.assertTrue(np.any(np.isnan(luminosity)))
        rcParams.update("cloudy","bounds_error",True)
        self.assertRaises(ValueError,self.CLOUDY.interpolate,name,metallicity,\
                              densityHydrogen,ionizingFluxHydrogen,\
                              ionizingFluxHeliumToHydrogen,\
                              ionizingFluxOxygenToHydrogen)



        print("TEST COMPLETE")
        print("\n")
        return

