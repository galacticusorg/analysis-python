#! /usr/bin/env python

import sys,os,fnmatch,glob,shutil
import numpy as np
import requests
import tarfile
from ..fileFormats.hdf5 import HDF5
from ..data import GalacticusData
from ..utils.progress import Progress

def getSSPDataFiles(chunks=1024,forceDownload=False):
    """
    getSSPDataFiles(): Function to download files containing Synthetic Stellar Population (SSP) models.

    USAGE: getSSPDataFiles([path=None],[chunks=1024],[extract=True],[forceDownload=False])

       INPUTS 
            path          -- Path to datasets repository. If None, will search for path in environment 
                             variables (stored as 'GALACTICUS_DATASETS').  [Default=None]
            chunks        -- If downloading the SSP files, specify the size of the chunks in which to
                             stream the download. [Default=1024]
            forceDownload -- Re-download the SSP file and overwrite existing copy. [Default=False]

    """
    # Set location of Galacticus dataset repository
    DATA = GalacticusData()
    spsDir = DATA.dynamic+"stellarPopulations/"
    if not os.path.exists(spsDir):
        os.makedirs(spsDir)
    # Location of file to download
    url = "http://users.obs.carnegiescience.edu/abenson/galacticus/data/Galacticus_SSP_Data.tar.bz2"
    # Set path to output
    outfile = spsDir+url.split("/")[-1]    
    # Download file if necessary
    if not os.path.exists(outfile) or forceDownload:
        print("Downloading SSP files...")
        # Open URL
        RESPONSE = requests.get(url, stream=True)    
        # Get length of content to initialize progress bar
        contentLength = int(RESPONSE.headers["Content-Length"])
        numberChunks = int(np.ceil(float(contentLength)/float(chunks)))    
        PROG = Progress(numberChunks)    
        # Open output file
        OUT = open(outfile, "wb")
        # Iterate over file content writing chunks to file
        for chunk in RESPONSE.iter_content(chunk_size=chunkSize):
            if chunk:  
                OUT.write(chunk)
            PROG.increment()
            PROG.print_status_line()
        # Close file
        OUT.close()    
    # Extract files from tar file
    print("Extracting SSP files...")
    TAR = tarfile.open(outfile)
    TAR.extractall(path=spsDir)
    for ifile in glob.glob(spsDir+"/data/stellarPopulations/*.hdf5"):
        name = ifile.split("/")[-1]
        shutil.move(ifile,spsDir+name)
    shutil.rmtree(spsDir+"/data")
    return
    

class SyntheticStellarPopulation(object):
    """
    SyntheticStellarPopulation: class to store SSP model information.

      Attributes:
    
             file : Name of SSP file.
             information: Dictionary containing reference for SSP model.
             wavelengths: Array of wavelengths for which spectra are specified.
             metallicites: Array of metallicities for which spectra are specified.
             ages: Array of ages for which spectra are specified.
             spectra: Array of spectra as function of wavelength, age and metallicity.

    """
    
    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initialize attributes
        self.file = None
        self.information = None
        self.wavelengths = None
        self.metallicities = None
        self.ages = None
        self.spectra = None
        self.imf = None
        return

    def wavelengthResolution(self,wavelength):        
        """
        wavelengthResolution(): Return the wavelength resolution of the SSP at a specified wavelength.

        USAGE: resolution = SyntheticStellarPopulation().wavelengthResolution(wavelength)

            INPUT
               wavelength -- Wavelength at which to query SSP.

            OUTPUT
               resolution -- Resolution of the SSP model at the specified wavelength.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not np.logical_and(wavelength>=self.wavelengths[0],wavelength<=self.wavelengths[-1]):
            raise IndexError(funcname+"(): specified wavelength is outside wavelength range of SSP model.")        
        diff = self.wavelengths - wavelength
        mask = diff > 0.0
        upp = self.wavelengths[mask].min()
        mask = diff <= 0.0
        low = self.wavelengths[mask].max()
        return upp-low
    
        
class GalacticusSyntheticStellarPopulations(object):
    """
    GalacticusSyntheticStellarPopulations: class to load SSP models.

    USAGE: GSSP = GalacticusSyntheticStellarPopulations([path=None][chunks=1024]

       INPUT
            path    -- Path to datasets repository. If None, will search for path in environment 
                       variables (stored as 'GALACTICUS_DATASETS').  [Default=None]
            chunks  -- If downloading the SSP files, specify the size of the chunks in which to
                       stream the download. [Default=1024]

    """
    def __init__(self,chunks=1024):
        DATA = GalacticusData()
        self.path = DATA.dynamic+"stellarPopulations/"
        self.models = glob.glob(self.path+"*.hdf5")
        if len(self.models) == 0:
            getSSPDataFiles(path=path,chunks=chunks,forceDownload=False)        
            self.models = glob.glob(self.path+"*.hdf5")
        self.models = [mod.split("/")[-1] for mod in self.models]
        return

    def load(self,fileName):        
        """
        load(): Load an SSP model into an instance of the SyntheticStellarPopulation class.

        USAGE:  SSP = GalacticusSyntheticStellarPopulations().load(fileName)

           INPUT
             fileName -- name of the HDF5 file containing the SSP to load.
             
           OUTPUT
              SSP -- instance of the SyntheticStellarPopulation class.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if not os.path.exists(fileName) or fileName.split("/")[-1] not in self.models:
            raise IOError(funcname+"(): File '"+fileName+"' not found.")        
        # Initialize SSP class
        SSP = SyntheticStellarPopulation()
        # Open file
        F = HDF5(fileName,'r')
        SSP.file = fileName
        # Extract information if present
        if "source" in F.lsGroups("/"):
            SSP.information = F.readAttributes("/source")
        # Load spectra
        SSP.wavelengths = np.copy(np.array(F.fileObj["wavelengths"]))
        SSP.metallicities = np.copy(np.array(F.fileObj["metallicities"]))
        SSP.ages = np.copy(np.array(F.fileObj["ages"]))
        SSP.spectra = np.copy(np.array(F.fileObj["spectra"]))
        # Load IMF if present
        if "initialMassFunction" in F.lsGroups("/"):
            n = len(np.array(F.fileObj["initialMassFunction/mass"]))
            SSP.imf = np.zeros(n,dtype=[("mass",float),("imf",float)]).view(np.recarray)
            SSP.imf.mass = np.copy(np.array(F.fileObj["initialMassFunction/mass"]))
            SSP.imf.imf = np.copy(np.array(F.fileObj["initialMassFunction/initialMassFunction"]))
        F.close()
        return SSP

    

            

