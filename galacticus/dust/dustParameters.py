#! /usr/bin/env python

import sys
import re
import numpy as np
from .. import rcParams
from ..datasets import Dataset
from ..properties.manager import Property

@Property.register_subclass('dustParameters')
class DustParameters(Property):
    """
    DustParameters: Compute the dust parameters, A_V and R_V.

    Functions:
            matches(): Indicates whether specified dataset can be processed by this class.
            get(): Computes A_V and R_V at specified redshift.

    """    
    dustParametersRegEx = "^(?P<component>[^:]+)LuminositiesStellar(?P<redshiftString>:z(?P<redshift>[\d\.]+)):dust(?P<label>[^:]+):(?P<parameter>A|R)_V"

    def __init__(self,galaxies):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = galaxies
        return

    def matches(self,propertyName,redshift=None):
        """
        DustParameters.matches(): Returns boolean to indicate whether this class can process
                                 the specified property.

        USAGE: match =  DusParameters.matches(propertyName,[redshift=None])

          INPUTS
              propertyName -- Name of property to process.
              redshift     -- Redshift value to query Galacticus HDF5 outputs.                           

          OUTPUTS
              match        -- Boolean indicating whether this class can process 
                              this property.

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return re.search(self.dustParametersRegEx,propertyName)

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
        propertyMatch = re.search(self.dustParametersRegEx,propertyName)
        if not propertyMatch:
            raise RuntimeError(funcname+"(): Cannot process property '"+propertyName+"'.")
        component      = propertyMatch.group("component"     )
        redshiftLabel  = propertyMatch.group("redshiftString")
        label          = propertyMatch.group("label"         )
        parameter      = propertyMatch.group("parameter"     )
        # Build names of the attenuated and unattenuated luminosity datasets.
        unattenuatedVDatasetName = component+"LuminositiesStellar:V:rest"+redshiftLabel
        attenuatedVDatasetName   = unattenuatedVDatasetName+":dust"+label
        unattenuatedBDatasetName = component+"LuminositiesStellar:B:rest"+redshiftLabel
        attenuatedBDatasetName   = unattenuatedBDatasetName+":dust"+label
        # Retrieve the luminosities.
        propertyNames = [ attenuatedVDatasetName, unattenuatedVDatasetName ]
        if (parameter == "R"):
            propertyNames.extend([ attenuatedBDatasetName, unattenuatedBDatasetName ])
        properties = self.galaxies.get(redshift,properties=propertyNames)
        # Compute the required parameter.
        dustParameter       = Dataset(name=propertyName)
        nonZeroV     = properties[unattenuatedVDatasetName].data > 0.0
        AV           = np.zeros(properties[unattenuatedVDatasetName].data.shape)
        AV[nonZeroV] = -2.5*np.log10(properties[attenuatedVDatasetName].data[nonZeroV]/properties[unattenuatedVDatasetName].data[nonZeroV])
        if (parameter == "R"):
            nonZeroB      = properties[unattenuatedBDatasetName].data > 0.0
            RV            = np.zeros(properties[unattenuatedBDatasetName].data.shape)
            AB            = np.zeros(properties[unattenuatedBDatasetName].data.shape)
            AB[nonZeroB]  = -2.5*np.log10(properties[attenuatedBDatasetName].data[nonZeroB]/properties[unattenuatedBDatasetName].data[nonZeroB])
            colorExcess   = AB-AV
            nonZeroExcess = colorExcess != 0.0
            RV[nonZeroExcess] = AV[nonZeroExcess]/colorExcess[nonZeroExcess]
            dustParameter.data = RV
        else:
            dustParameter.data = AV
        return dustParameter
    
