#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import warnings
import unittest
from .. import rcParams
from ..datasets import Dataset
from ..properties.manager import Property
from ..Cloudy import CloudyTable
from ..filters.filters import GalacticusFilter
from ..constants import massSolar,luminositySolar,metallicitySolar
from ..constants import luminosityAB,erg
from ..constants import parsec,angstrom
from ..constants import mega,centi
from ..constants import Pi,speedOfLight
from ..constants import massAtomic,atomicMassHydrogen,massFractionHydrogen

@Property.register_subclass('emissionLineLuminosity')
class EmissionLineLuminosity(Property):
    
    def __init__(self,galaxies,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        self.verbose = verbose
        self.CLOUDY = CloudyTable()
        self.GALFIL = GalacticusFilter()
        return

    def lineInCloudyOutput(self,lineName):
        """
        EmissionLineLuminosity.lineInCloudyOutput: Returns boolean indicating whether specified
                                                   emission line can be found in CLOUDY output.

        USAGE:  result = EmissionLineLuminosity.lineInCloudyOutput(lineName)
        
          INPUTS
              lineName -- Name of emission line.

          OUTPUTS
              result   -- Boolean (T/F) indicating whether specified line is present.

        """
        return lineName in self.CLOUDY.listAvailableLines()

    def parseDatasetName(self,datasetName):
        """
        EmissionLineLuminosity.parseDatasetName: Parse an emission line luminosity 
                                                 dataset name.
        
        USAGE: SEARCH = EmissionLineLuminosity.parseDatasetName(propertyName)
        
             INPUTS
                propertyName -- Property name to parse.
                
             OUTPUTS
                SEARCH       -- Regex seearch (re.search) object or None if
                                propertyName cannot be parsed.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Construct search string to pass to regex
        searchString = "^(?P<component>disk|spheroid)LineLuminosity:"
        lines = "(?P<lineName>"+"|".join(self.CLOUDY.listAvailableLines())+")"
        searchString = searchString + lines + ":(?P<frame>rest|observed)"+\
            "(?P<filterName>:[^:]+)?"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            "(?P<recent>:recent)?$"
        return re.search(searchString,datasetName)
    
    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        EmissionLineLuminosity.matches: Returns boolean to indicate whether this 
                                        class can process the specified property.

        USAGE:  match = EmissionLineLuminosity.matches(propertyName,[redshift=None],
                                                       [raiseError=False])
                
          INPUTS 
               propertyName -- Name of property to process.
                   redshift -- Redshift value to query Galacticus HDF5 outputs. 
                               (Redundant in this particular case, but required 
                               for other properties.)
                raiseError  -- Raise error if property does not match. 
                               (Default = False)                

          OUTPUTS 
                match       -- Boolean indicating whether this class can process 
                               this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid emission line luminosity. "+\
                "Available emission lines: "+\
                ", ".join(self.CLOUDY.listAvailableLines())+"."
            raise RuntimeError(msg)
        return False

    def getContinuumLuminosityNames(self,propertyName):
        """
        EmissionLineLuminosity.getContinuumLuminosityNames: For specified emission line luminosity dataset,
                                                            return the name of the appopriate continuum
                                                            luminosity datasets.
        
        USAGE: LyName,HeName,OxName = EmissionLineLuminosity.getContinuumLuminosityNames(propertyName)
        
              INPUTS
                    propertyName -- Emission line dataset name
                    
              OUTPUTS
                    LyName       -- Lyman continuum luminosity dataset name
                    HeName       -- Helium continuum luminosity dataset name
                    OxName       -- Oxygen continuum luminosity dataset name
                    
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.matches(propertyName,raiseError=True)
        MATCH = self.parseDatasetName(propertyName)
        recent = ""
        if MATCH.group("recent"):
            recent = MATCH.group("recent")
        LymanName = MATCH.group("component")+"LymanContinuumLuminosity"+MATCH.group("redshiftString")+recent
        HeliumName = MATCH.group("component")+"HeliumContinuumLuminosity"+MATCH.group("redshiftString")+recent
        OxygenName = MATCH.group("component")+"OxygenContinuumLuminosity"+MATCH.group("redshiftString")+recent
        return LymanName,HeliumName,OxygenName

    def getContinuumLuminosities(self,propertyName,redshift):
        """
        EmissionLineLuminosity.getContinuumLuminosities: For specified emission line luminosity dataset,
                                                         return dictionary of appopriate continuum
                                                         luminosity datasets.
        
        USAGE: LUMINOSITIES = EmissionLineLuminosity.getContinuumLuminosities(propertyName,redshift)
        
              INPUTS
                    propertyName -- Emission line dataset name
                    redshift     -- Redshift to query Galacticus HDF5 outputs.
                    
              OUTPUTS
                    LUMINOSITIES -- Dictionary of Dataset() instances containing Lyman, Helium and 
                                    Oxygen continuuum luminosities. For any that are missing
                                    those dictionary entries will be set to None.
                    
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        LymanName,HeliumName,OxygenName = self.getContinuumLuminosityNames(propertyName)
        names = [LymanName,HeliumName,OxygenName]
        return self.galaxies.get(redshift,properties=names)

    def getIonizingFluxHydrogen(self,LyLuminosity):        
        """
        EmissionLineLuminosity.getIonizingFluxHydrogen(): Compute the Lyman ionizing flux from the supplied
                                                          Lyman continuuum luminosity.

        USAGE: flux = EmissionLineLuminosity.getIonizingFluxHydrogen(luminosity)

           INPUTS
              luminosity -- Lyman continuum luminosity
              
           OUTPUTS
              flux       -- log10 of Lyman ionizing flux.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        ionizingFluxHydrogen = np.copy(LyLuminosity)
        np.place(ionizingFluxHydrogen,ionizingFluxHydrogen==0.0,np.nan)
        return np.log10(ionizingFluxHydrogen)+50.0
    
    def getIonizingFluxRatio(self,LyLuminosity,XLuminosity):
        """
        EmissionLineLuminosity.getIonizingFluxRatio(): Compute the ratio of Helium or Oxygen ionizing flux 
                                                       to Lyman ionizing flux.

        USAGE: ratio = EmissionLineLuminosity.getIonizingFluxRatio(LymanLuminosity,XLuminosity)

           INPUTS
              LymanLuminosity -- Lyman continuum luminosity.
              XLuminosity     -- Helium or Oxygen luminosity.
              
           OUTPUTS
              ratio           -- log10 of ionizing flux ratio..

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        LymanFlux = np.copy(LyLuminosity)        
        np.place(LymanFlux,LymanFlux==0.0,np.nan)
        XFlux = np.copy(XLuminosity)
        np.place(XFlux,XFlux==0.0,np.nan)
        return np.log10(XFlux/LymanFlux)

    def getMassHIIRegions(self):
        """
        EmissionLineLuminosity.getMassHIIRegion(): Return the mass of HII regions, as stored in 
                                                   the configuration parameters. Default value 
                                                   is 7.5e3 Msol.
                                                   
        USAGE: mass = EmissionLineLuminosity.getMassHIIRegion()

           OUTPUTS
               mass -- Mass of HII regions in Solar masses.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return rcParams.getfloat("emissionLine","massHIIRegion",fallback=7.5e3)
    
    def getLifetimeHIIRegions(self):
        """
        EmissionLineLuminosity.getLifetimeHIIRegion(): Return the lifetime of HII regions, as stored in 
                                                       the configuration parameters. Default value is 
                                                       1.0e-3 Gyr.
                                                   
        USAGE: mass = EmissionLineLuminosity.getLifetimeHIIRegion()

           OUTPUTS
               mass -- Lifetime of HII regions in Gyrs.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return rcParams.getfloat("emissionLine","lifetimeHIIRegion",fallback=1.0e-3)

    def getNumberHIIRegions(self,redshift,component):
        """
        EmissionLineLuminosity.getNumberHIIRegions(): Return number of HII regions in for specified
                                                      galaxy component at specified redshift.

        USAGE: number = EmissionLineLuminosity.getNumberHIIRegions(redshift,component)
        
           INPUTS
               redshift  -- Redshift value to query Galacticus HDF5 outputs.
               component -- String indicating component to compute number for. String
                            be either 'disk' or 'spheroid'.

           OUTPUT
               number   -- Number of HII regions.      
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): Component '"+component+"' not recognized. "+\
                                 "Should be either 'disk' or 'spheroid'.")
        sfrName = component+"StarFormationRate"
        GALS = self.galaxies.get(redshift,properties=[sfrName])
        massHIIRegion = self.getMassHIIRegions()
        lifetimeHIIRegion = self.getLifetimeHIIRegions()
        return GALS[sfrName].data*lifetimeHIIRegion/massHIIRegion
        
    def getHydrogenGasDensity(self,redshift,component):
        """
        EmissionLineLuminosity.getHydrogenGasDensity(): Compute Hydrogen gas density at given redshift
                                                        for specified galaxy component.

        USAGE: density = EmissionLineLuminosity.getHydrogenGasDensity(redshift,component)

           INPUTS
               redshift  -- Redshift value to query Galacticus HDF5 outputs.
               component -- String indicating component to compute number for. String
                            be either 'disk' or 'spheroid'.
        
           OUTPUTS
               density   -- Numpy array of Hydrogen gas density.
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): Component '"+component+"' not recognized. "+\
                                 "Should be either 'disk' or 'spheroid'.")
        # Extract gas mass and galaxy radius
        gas = component+"MassGas"
        radius = component+"Radius"
        GALS = self.galaxies.get(redshift,properties=[gas,radius])       
        # Compute volume in cm^3
        volume = np.copy((GALS[radius].data*mega*parsec/centi)**3)
        np.place(volume,volume==0.0,np.nan)
        # Compute gas mass in kg
        mass = np.copy(GALS[gas].data)*massSolar
        np.place(mass,mass==0.0,np.nan)
        # Clear GALS from memory
        del GALS
        # Compute density in kg/cm^3        
        density = np.copy(mass/volume)
        density *= massFractionHydrogen
        density /= (4.0*Pi*massAtomic*atomicMassHydrogen)
        density = np.log10(density)
        # Clear memory from unnecessary variables
        del mass,volume
        return density

    def getLuminosityMultiplier(self,propertyName,redshift):
        """
        EmissionLineLuminosity.getLuminosityMultiplier(): If emission line is under a broadband filter,
                                                          compute the multiplicative factor to convert 
                                                          line luminosity to broadband luminosity in 
                                                          AB units. If line is not under a filter, i.e.
                                                          no filter is specified in the properrty name,
                                                          then the multiplication factor is unity.

        USAGE: multiplier = EmissionLineLuminosity.getLuminosityMultiplier(propertyName,redshift)

          INPUTS
               propertyName -- Property name to compute multiplier for. Should be a
                               valid emission line luminosity dataset name.
               redshift     -- Redshift value to query Galacticus HDF5 outputs.
         

          OUTPUTS
               multiplier   -- Multiplication factor. Equal to unity if no filter
                               specified.


        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.matches(propertyName,raiseError=True)
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)
        # Check if filter name provided
        filterName = MATCH.group('filterName')
        # Exit if no filter provided
        if filterName is None:
            return 1.0
        # Compute multiplier for filter
        # Load Filter instance
        FILTER = self.GALFIL.load(filterName.replace(":",""))
        # Get the redshift of the galaxies
        GALS = self.galaxies.get(redshift,properties=["redshift"])
        # Extract frame (rest or observed)
        frame = MATCH.group('frame').replace(":","")
        # Get line wavelength
        lineWavelength = self.CLOUDY.getWavelength(MATCH.group('lineName'))
        if frame == "observed":            
            lineWavelength *= (1.0+GALS['redshift'].data)
        else:
            lineWavelength *= np.ones_like(GALS['redshift'].data)
        # Interpolate the transmission to the line wavelength
        multiplier = FILTER.interpolate(lineWavelength)
        # Compute the multiplicative factor to convert line
        # luminosity to luminosity in AB units in the filter
        multiplier /= FILTER.integrate()
        # Galacticus defines observed-frame luminosities
        # by simply redshifting the galaxy spectrum without
        # changing the amplitude of F_nu (i.e. the compression of
        # the spectrum into a smaller range of frequencies is not
        # accounted for). For a line, we can understand how this
        # should affect the luminosity by considering the line as
        # a Gaussian with very narrow width (such that the full
        # extent of the line always lies in the filter). In this
        # case, when the line is redshifted the width of the
        # Gaussian (in frequency space) is reduced, while the
        # amplitude is unchanged (as, once again, we are not
        # taking into account the compression of the spectrum into
        # the smaller range of frequencies). The integral over the
        # line will therefore be reduced by a factor of (1+z) -
        # this factor is included in the following line. Note
        # that, when converting this observed luminosity into an
        # observed flux a factor of (1+z) must be included to
        # account for compression of photon frequencies (just as
        # with continuum luminosities in Galacticus) which will
        # counteract the effects of the 1/(1+z) included below.
        if frame == "observed":
            multiplier /= (1.0+GALS["redshift"].data)
        else:
            lineWavelength *= np.ones_like(GALS['redshift'].data)
        return multiplier

    def get(self,propertyName,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.matches(propertyName,raiseError=True)
        # Extract information from property name
        MATCH = self.parseDatasetName(propertyName)        
        # Get continuum luminosity names
        LymanName,HeliumName,OxygenName = self.getContinuumLuminosityNames(propertyName)
        # Extract continuum luminosities
        FLUXES = self.getContinuumLuminosities(propertyName)
        # If any luminosities are missing, unable to compute emission lines so return 'None' instance.
        if any([FLUXES[name] is None for name in FLUXES.keys()]):
            warnings.warn(funcname+"(): Unable to compute emission line luminosity as one of the "+\
                              "continuum luminosities is missing. Returning None instance.")
            return None
        # Get galaxy properties for calculation
        # i) Hydrogen gas density
        hydrogenGasDensity = self.getHydrogenGasDensity(redshift,MATCH.group('component'))
        # ii) Metallicity
        metals = MATCH.group('component')+"GasMetallicity"
        GALS = self.galaxies.get(redshift,properties=[metals])
        metallicity = np.copy(GALS[metals].data)
        del GALS
        # iii) Ionizing Hydrogen flux
        ionizingFluxHydrogen = self.getIonizingFluxHydrogen(FLUXES[LymanName].data,redshift)
        #      Convert the hydrogen ionizing luminosity to be per HII region
        numberHIIRegion = self.getNumberHIIRegions(redshift,MATCH.group('component'))
        np.place(numberHIIRegion,numberHIIRegion==0.0,np.nan)
        ionizingFluxHydrogen -= np.log10(numberHIIRegion)
        # iv) Luminosity ratios He/H and Ox/H 
        ionizingFluxHeliumToHydrogen = self.getIonizingFluxRatio(FLUXES[LymanName].data,FLUXES[HeliumName].data)
        ionizingFluxOxygenToHydrogen = self.getIonizingFluxRatio(FLUXES[LymanName].data,FLUXES[OxygenName].data)
        # Create Dataset() instance        
        DATA = Dataset(name=propertyName)
        attr = {"unitsInSI":luminositySolar}
        attr["massHIIRegion"] = self.getMassHIIRegions()
        attr["lifetimeHIIRegion"] = self.getLifetimeHIIRegions()
        DATA.attr = attr
        # Pass properties to CloudyTable() class for interpolation        
        DATA.data = np.copy(self.CLOUDY.interpolate(MATCH.group("lineName"),metallicity,densityHydrogen,\
                                                        ionizingFluxHydrogen,ionizingFluxHeliumToHydrogen,\
                                                        ionizingFluxOxygenToHydrogen))
        # Clear memory
        del metallicity,densityHydrogen,ionizingFluxHydrogen
        del ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen
        # Get luminosity multiplier
        luminosityMultiplier = self.getLuminosityMultiplier()
        # Convert units of luminosity 
        DATA.data *= (luminosityMultiplier*numberHIIRegion*erg/luminositySolar)
        return DATA



class UnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from ..galaxies import Galaxies
        from ..io import GalacticusHDF5
        from ..data import GalacticusData
        from shutil import copyfile
        # Locate the dynamic version of the galacticus.snapshotExample.hdf5 file.
        DATA = GalacticusData(verbose=False)
        self.snapshotFile = DATA.searchDynamic("galacticus.snapshotExample.hdf5")
        self.removeExample = False
        # If the file does not exist, create a copy from the static version.
        if self.snapshotFile is None:
            self.snapshotFile = DATA.dynamic+"/examples/galacticus.snapshotExample.hdf5"
            self.removeExample = True
            if not os.path.exists(DATA.dynamic+"/examples"):
                os.makedirs(DATA.dynamic+"/examples")
            copyfile(DATA.static+"/examples/galacticus.snapshotExample.hdf5",self.snapshotFile)
        # Initialize the IonizingContinuum class.
        GH5 = GalacticusHDF5(self.snapshotFile,'r')
        GALS = Galaxies(GH5Obj=GH5)
        self.LINES = EmissionLineLuminosity(GALS)
        return

    @classmethod
    def tearDownClass(self):
        # Clear memory and close/delete files as necessary.
        self.LINES.galaxies.GH5Obj.close()
        del self.LINES
        if self.removeExample:
            os.remove(self.snapshotFile)
        return

    def testMatches(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.matches() function")
        for line in self.LINES.CLOUDY.listAvailableLines():
            for component in ["disk","spheroid"]:
                name = component+"LineLuminosity:"+line+":rest:z1.000"                
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineLuminosity:"+line+":observed:SDSS_r:z1.000"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineLuminosity:"+line+":observed:z1.000:recent"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineLuminosity:"+line+":rest:SDSS_g:z1.000:recent"
                self.assertTrue(self.LINES.matches(name))
                name = component+"LineLuminosity:"+line
                self.assertFalse(self.LINES.matches(name))
                name = component+"LineLuminosity:"+line+":rest:z1.000:dustAtlas"                
                self.assertFalse(self.LINES.matches(name,raiseError=False))
                self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
                name = component+"Luminosity:"+line+":rest:z1.000"                
                self.assertFalse(self.LINES.matches(name,raiseError=False))
                self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
                name = component+"LineLuminosity:z1.000:"+line
                self.assertFalse(self.LINES.matches(name,raiseError=False))
                self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
                name = component+"LineLuminosity:"+line+":z1.000"                
                self.assertFalse(self.LINES.matches(name,raiseError=False))
                self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
                name = component+"LineLuminosity:SDSS_r:rest:"+line+":z1.000"                
                self.assertFalse(self.LINES.matches(name,raiseError=False))
                self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
            name = "totalLineLuminosity:"+line+":rest:z1.000"                
            self.assertFalse(self.LINES.matches(name,raiseError=False))
            self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
            self.assertFalse(self.LINES.matches("basicMass",raiseError=False))
            self.assertRaises(RuntimeError,self.LINES.matches,"basicMass",raiseError=True)
        name = "diskLineLuminosity:notAnEmissionLine:rest:z1.000"                
        self.assertFalse(self.LINES.matches(name,raiseError=False))
        self.assertRaises(RuntimeError,self.LINES.matches,name,raiseError=True)
        print("TEST COMPLETE")
        print("\n")
        return

    def testLineInCloudyOutput(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.lineInCloudyOutput() function")
        for line in self.LINES.CLOUDY.listAvailableLines():
            self.assertTrue(self.LINES.lineInCloudyOutput(line))
        self.assertFalse(self.LINES.lineInCloudyOutput("notAnEmissionLine"))
        print("TEST COMPLETE")
        print("\n")
        return
        
    def testGetContinuumLuminosityNames(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getContinuumLuminosityNames() function")
        name = "diskLineLuminosity:balmerAlpha6563:rest:z1.000"
        Ly,He,O = self.LINES.getContinuumLuminosityNames(name)
        self.assertEqual(Ly,"diskLymanContinuumLuminosity:z1.000")
        self.assertEqual(He,"diskHeliumContinuumLuminosity:z1.000")
        self.assertEqual(O,"diskOxygenContinuumLuminosity:z1.000")
        name = "spheroidLineLuminosity:balmerAlpha6563:observed:z1.000:recent"
        Ly,He,O = self.LINES.getContinuumLuminosityNames(name)
        self.assertEqual(Ly,"spheroidLymanContinuumLuminosity:z1.000:recent")
        self.assertEqual(He,"spheroidHeliumContinuumLuminosity:z1.000:recent")
        self.assertEqual(O,"spheroidOxygenContinuumLuminosity:z1.000:recent")
        name = "spheroidLineLuminosity:balmerAlpha6563:observed"
        self.assertRaises(RuntimeError,self.LINES.getContinuumLuminosityNames,name)
        name = "totalLineLuminosity:balmerAlpha6563:observed:z1.000"
        self.assertRaises(RuntimeError,self.LINES.getContinuumLuminosityNames,name)
        name = "diskLineLuminosity:balmerAlpha6563"
        self.assertRaises(RuntimeError,self.LINES.getContinuumLuminosityNames,name)
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetContinuumLuminosities(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getContinuumLuminosities() function")
        redshift = 1.0
        OUT = self.LINES.galaxies.GH5Obj.selectOutput(redshift)
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)        
        names = ["spheroidLineLuminosity:balmerAlpha6563:rest:"+zStr,\
                     "diskLineLuminosity:balmerBeta4861:observed:"+zStr]
        for name in names:
            Ly,He,O = self.LINES.getContinuumLuminosityNames(name)        
            LUMS = self.LINES.getContinuumLuminosities(name,redshift)
            for name in [Ly,He,O]:
                self.assertIsNotNone(LUMS[name])
        print("TEST COMPLETE")
        print("\n")
        return

    def testGetIonizingFluxHydrogen(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getIonizingFluxHydrogen() function")
        luminosity = np.random.rand(50)*1.0e20 + 1.0e3
        ionizingFluxHydrogen = np.copy(luminosity)
        np.place(ionizingFluxHydrogen,ionizingFluxHydrogen==0.0,np.nan)
        ionizingFluxHydrogen = np.log10(ionizingFluxHydrogen)+50.0
        result = self.LINES.getIonizingFluxHydrogen(luminosity)
        diff = np.fabs(result-ionizingFluxHydrogen)
        [self.assertLessEqual(d,1.0e06) for d in diff]
        np.place(luminosity,np.random.rand(50)<0.1,0.0)
        result = self.LINES.getIonizingFluxHydrogen(luminosity)
        for i in range(len(result)):
            self.assertEqual(luminosity[i]==0.0,np.isnan(result[i]))
        print("TEST COMPLETE")
        print("\n")
        return            
        
    def testGetIonizingFluxRatio(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getIonizingFluxRatio() function")
        lyman = np.random.rand(50)*1.0e20 + 1.0e3
        helium = np.random.rand(50)*1.0e20 + 1.0e3
        ratio = np.log10(helium/lyman)
        result = self.LINES.getIonizingFluxRatio(lyman,helium)
        diff = np.fabs(ratio-result)
        [self.assertLessEqual(d,1.0e-6) for d in diff]
        np.place(lyman,np.random.rand(50)<0.1,0.0)
        np.place(helium,np.random.rand(50)<0.1,0.0)
        mask = np.logical_or(lyman==0.0,helium==0.0)
        result = self.LINES.getIonizingFluxRatio(lyman,helium)
        for i in range(len(result)):
            self.assertEqual(mask[i],np.isnan(result[i]))
        print("TEST COMPLETE")
        print("\n")
        return            

    def testGetMassHIIRegions(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getMassHIIRegions() function")
        mass = np.random.rand(1)[0]*1.0e3
        rcParams.update("emissionLine","massHIIRegion",mass)
        diff = np.fabs(mass-self.LINES.getMassHIIRegions())
        self.assertLessEqual(diff,1.0e-6)
        print("TEST COMPLETE")
        print("\n")
        return            

    def testGetLifetimeHIIRegions(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getLifetimeHIIRegions() function")
        lifetime = np.random.rand(1)[0]*1.0e-3
        rcParams.update("emissionLine","lifetimeHIIRegion",lifetime)
        diff = np.fabs(lifetime-self.LINES.getLifetimeHIIRegions())
        self.assertLessEqual(diff,1.0e-6)
        print("TEST COMPLETE")
        print("\n")
        return            

    def testGetNumberHIIRegions(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getNumberHIIRegions() function")
        redshift = 1.0
        OUT = self.LINES.galaxies.GH5Obj.selectOutput(redshift)
        for component in ["disk","spheroid"]:
            sfr = np.array(OUT["nodeData/"+component+"StarFormationRate"])
            n = sfr*self.LINES.getLifetimeHIIRegions()
            n /= self.LINES.getMassHIIRegions()
            N = self.LINES.getNumberHIIRegions(redshift,component)
            diff = np.fabs(n-N)
            [self.assertLessEqual(d,1.0e-6) for d in diff]
        self.assertRaises(ValueError,self.LINES.getNumberHIIRegions,redshift,"total")
        print("TEST COMPLETE")
        print("\n")
        return            

    def testGetHydrogenGasDensity(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getHydrogenGasDensity() function")
        redshift = 1.0
        OUT = self.LINES.galaxies.GH5Obj.selectOutput(redshift)
        for comp in ["disk","spheroid"]:
            gas = np.array(OUT["nodeData/"+comp+"MassGas"])
            radius = np.array(OUT["nodeData/"+comp+"Radius"])
            volume = np.copy((radius*mega*parsec/centi)**3)
            np.place(volume,volume==0.0,np.nan)
            mass = np.copy(gas)*massSolar
            np.place(mass,mass==0.0,np.nan)
            density = np.copy(mass/volume)
            density *= massFractionHydrogen
            density /= (4.0*Pi*massAtomic*atomicMassHydrogen)
            density = np.log10(density)
            result = self.LINES.getHydrogenGasDensity(redshift,comp)
            for d,r in zip(density,result):
                self.assertEqual(np.isnan(d),np.isnan(r))
                if not np.isnan(d):
                    diff = np.fabs(d-r)
                    self.assertLessEqual(diff,1.0e-6)
        self.assertRaises(ValueError,self.LINES.getHydrogenGasDensity,\
                              redshift,"total")
        print("TEST COMPLETE")
        print("\n")
        return            

    def testGetLuminosityMultiplier(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("UNIT TEST: Emission Line Luminosity: "+funcname)
        print("Testing EmissionLineLuminosity.getLuminosityMultiplier() function")
        redshift = 1.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        for filterName in ["SDSS_r","SDSS_g"]:
            for frame in ["rest","observed"]:
                for line in ["balmerAlpha6563","balmerBeta4861"]:
                    name = "diskLineLuminosity:"+line+":"+frame+":"+filterName+":"+zStr            
                    FILTER = self.LINES.GALFIL.load(filterName)
                    GALS = self.LINES.galaxies.get(redshift,properties=["redshift"])                    
                    lineWavelength = self.LINES.CLOUDY.getWavelength(line)
                    if frame == "observed":
                        lineWavelength *= (1.0+GALS['redshift'].data)
                    else:
                        lineWavelength *= np.ones_like(GALS['redshift'].data)
                    multiplier = FILTER.interpolate(lineWavelength)
                    multiplier /= FILTER.integrate()
                    if frame == "observed":
                        multiplier /= (1.0+GALS["redshift"].data)
                    else:
                        lineWavelength *= np.ones_like(GALS['redshift'].data)
                    result = self.LINES.getLuminosityMultiplier(name,redshift)
                    diff = np.fabs(multiplier-result)
                    [self.assertLessEqual(d,1.0e-6) for d in diff]
        name = "diskLineLuminosity:balmerAlpha6563:rest:"+zStr
        result = self.LINES.getLuminosityMultiplier(name,redshift)
        self.assertEqual(result,1.0)
        redshift = 2.0
        zStr = self.LINES.galaxies.GH5Obj.getRedshiftString(redshift)
        name = "spheroidLineLuminosity:balmerAlpha6563:observed:"+zStr
        result = self.LINES.getLuminosityMultiplier(name,redshift)
        self.assertEqual(result,1.0)
        name = "diskLineLuminosity:balmerAlpha6563:SDSS_r:"+zStr
        self.assertRaises(RuntimeError,self.LINES.getLuminosityMultiplier,\
                              name,redshift)
        name = "diskLineLuminosity:balmerAlpha6563:rest:SDSS_h:"+zStr
        self.assertRaises(RuntimeError,self.LINES.getLuminosityMultiplier,\
                              name,redshift)
        print("TEST COMPLETE")
        print("\n")
        return            
