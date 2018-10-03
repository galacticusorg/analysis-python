#! /usr/bin/env python

import sys
import re
import numpy as np
import unittest
from .. import rcParams
from ..datasets import Dataset
from ..utils import match_dimensions
from ..properties.manager import Property
from .screens.manager import ScreenLaw

SCREENS = ScreenLaw()


@Property.register_subclass('dustParameters')
class DustParameters(Property):
    """
    DustParameters: Compute the dust parameters, A_V and R_V.

    Functions:
            parseDatasetName(): Parse a dust parameter dataset.
            matches(): Indicates whether specified dataset can be processed by this class.
            getAttenuationParameter(): Compute A_X given attenuated and unattenuated 
                                       X-band luminosities.
            getReddeningParameter(): Compute R_V given attenuated and unattenuated 
                                       V-band and B-band luminosities.
            get(): Computes A_V and R_V at specified redshift.

    """    
    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def parseDatasetName(self,propertyName):
        """
        DustParameters.parseDatasetName: Parse a dust parameters dataset.

        USAGE: SEARCH = DustParameters.parseDatasetName(propertyName)

             INPUTS
                propertyName -- Property name to parse.

             OUTPUTS

                SEARCH       -- Regex search (re.search) object or None if
                                propertyName cannot be parsed.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dustRegex = ":dust(?P<label>"+"|".join(SCREENS.laws.keys())+\
            "|Atlas|Compendium|CharlotFall2000)"
        searchString = "^(?P<component>[^:]+)LuminositiesStellar"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
            dustRegex+":(?P<parameter>A|R)_V"
        #searchString = "^(?P<component>[^:]+)LuminositiesStellar"+\
        #    "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
        #    ":dust(?P<label>[^:]+):(?P<parameter>A|R)_V"
        MATCH = re.search(searchString,propertyName)
        return MATCH

    def matches(self,propertyName,redshift=None,raiseError=False):
        """
        DustParameters.matches(): Returns boolean to indicate whether this class can process
                                 the specified property.

        USAGE: match =  DustParameters.matches(propertyName,[redshift=None],[raiseError=False])

          INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.                           
              raiseError   -- Raise error if property does not match.
                              (Default = False)  

          OUTPUTS
              match        -- Boolean indicating whether this class can process 
                              this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = self.parseDatasetName(propertyName)
        if MATCH is not None:
            return True
        if raiseError:
            msg = funcname+"(): Specified property '"+propertyName+\
                "' is not a valid dust parameter dataset. "
            raise RuntimeError(msg)
        return False
    
    def getAttenuationParameter(self,attenL,unattenL):                
        """
        DustParameters.getAteenuationParameter(): Compute attenuation parameter.
        
        USAGE:  A = DustParameters.getAttenuationParameter(attenL,unattenL)
                
           INPUTS           
                attenL   -- Attenuated luminosity.
                unattenL -- Un-attenuated luminosity.
           
           OUTPUT
                A        -- Numpy array of attenuation dust parameters.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not match_dimensions(attenL,unattenL):
            msg = funcname+"(): attenuated and unattenuated luminosity arrays "+\
                "have different dimensions."
            raise ValueError(msg)
        nonZero     = unattenL > 0.0        
        A           = np.ones_like(unattenL)*np.nan
        A[nonZero] = -2.5*np.log10(attenL[nonZero]/unattenL[nonZero])
        return A

    def getReddeningParameter(self,attenV,unattenV,attenB,unattenB):
        """
        DustParameters.getReddening(): Compute reddening parameter.
        
        USAGE:  R = DustParameters.getReddeningParameter(attenV,unattenV,
                                                         attenB,unattenB)
                
           INPUTS           
                attenV   -- Attenuated V-band luminosity.
                unattenV -- Un-attenuated V-band luminosity.
                attenB   -- Attenuated B-band luminosity.
                unattenB -- Un-attenuated B-band luminosity.
           
           OUTPUT
                R        -- Numpy array of reddening parameters.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not match_dimensions(attenV,unattenV,attenB,unattenB):
            msg = funcname+"(): attenuated and unattenuated luminosity arrays "+\
                "have different dimensions."
            raise ValueError(msg)
        AV = self.getAttenuationParameter(attenV,unattenV)
        AB = self.getAttenuationParameter(attenB,unattenB)
        colorExcess = AB - AV
        RV = np.ones_like(AV)*np.nan
        mask = colorExcess>0.0
        RV[mask] = AV[mask]/colorExcess[mask]
        return RV
                
    def get(self,propertyName,redshift):        
        """
        DustParameters.get(): Compute dust model parameters for specified redshift.
        
        USAGE:  DATA = DustParameters.get(propertyName,redshift)
                
           INPUTS
           
                propertyName -- Name of property to compute.
                redshift     -- Redshift value to query Galacticus HDF5 outputs.
           
           OUTPUT
                DATA         -- Instance of galacticus.datasets.Dataset() class containing 
                                computed galaxy information.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        assert(self.matches(propertyName,raiseError=True))
        # Extract information from property name
        propertyMatch = self.parseDatasetName(propertyName)
        component      = propertyMatch.group("component"     )
        redshiftLabel  = propertyMatch.group("redshiftString")
        label          = propertyMatch.group("label"         )
        parameter      = propertyMatch.group("parameter"     )
        # Build names of the attenuated and unattenuated luminosity datasets.
        unattenuatedVDatasetName = component+"LuminositiesStellar:Buser_V:rest"+redshiftLabel
        attenuatedVDatasetName   = unattenuatedVDatasetName+":dust"+label
        unattenuatedBDatasetName = component+"LuminositiesStellar:Buser_B:rest"+redshiftLabel
        attenuatedBDatasetName   = unattenuatedBDatasetName+":dust"+label
        # Retrieve the luminosities.
        propertyNames = [ attenuatedVDatasetName, unattenuatedVDatasetName ]
        if (parameter == "R"):
            propertyNames.extend([ attenuatedBDatasetName, unattenuatedBDatasetName ])
        PROPS = self.galaxies.get(redshift,properties=propertyNames)
        # Compute the required parameter.
        DATA = Dataset(name=propertyName)
        if (parameter == "A"):
            DATA.data = np.copy(self.getAttenuationParameter(PROPS[attenuatedVDatasetName].data,\
                                                         PROPS[unattenuatedVDatasetName].data))
        elif (parameter == "R"):
            DATA.data = np.copy(self.getReddeningParameter(PROPS[attenuatedVDatasetName].data,\
                                                       PROPS[unattenuatedVDatasetName].data,\
                                                       PROPS[attenuatedBDatasetName].data,\
                                                       PROPS[unattenuatedBDatasetName].data))
        else:
            raise ValueError(funcname+"(): Parameter '"+parameter\
                                 +"'not recognized. Should be A or R.")
        del PROPS
        return DATA
    
