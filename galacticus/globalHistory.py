#! /usr/bin/env python

import sys
import numpy as np
from .io import GalacticusHDF5
from .datasets import Dataset


class GlobalHistory(object):
        
    @classmethod
    def get(cls,GH5):
        if "globalHistory" not in GH5.lsGroups("/"):
            return None
        dsets = GH5.lsDatasets("/globalHistory")
        dtype = [(name,float) for name in dsets+["historyRedshift"]]
        size = len(GH5.fileObj["/globalHistoryhistoryExpansion"])
        GLOBAL = {}
        for dset in dsets:
            data = GH5.readDataset("/globalHistory/"+dset)
            attr = GH5.readAttributes("/globalHistory/"+dset)
            GLOBAL[dset] = Dataset(name=dset,data=data,attr=attr)
        name = "historyRedshift"
        GLOBAL[name] = Dataset(name=name,data=1.0/(GLOBAL["historyExpansion"].data-1.0))
        return GLOBAL
        



def getGlobalHistory(GH5Obj,required=None,unitsInSI=False):    
    """
    getGlobalHistory(): Extract the global history information, if present, from 
                        a Galacticus HDF5 file.

    USAGE:  history = getGlobalHistory(GH5Obj,[required=None],[unitsInSI=False])
    
       INPUTS
             GH5Obj    -- A GalacticusHDF5 class object.
             required  -- List of datasets to extract. If None, will extract all
                          available datasets in global history sub-directory.
                          (Default=None)
             unitsInSI -- Return datasets in SI units. (Default=False)

       OUTPUTS
             history   -- Structured Numpy array containing history information, or
                          'None' if global history information is not present.

    """
    if "globalHistory" not in GH5Obj.fileObj["/"].keys():
        return None
    globalHistory = GH5Obj.fileObj["globalHistory"]
    allprops = globalHistory.keys() + ["historyRedshift"]
    if required is None:
        required = allprops
    else:
        required = set(required).intersection(allprops)
    epochs = len(np.array(globalHistory["historyExpansion"]))
    dtype = np.dtype([ (str(p),np.float) for p in required ])
    history = np.zeros(epochs,dtype=dtype)
    for p in history.dtype.names:
        if p is "historyRedshift":
            history[p] = np.copy((1.0/np.array(globalHistory["historyExpansion"]))-1.0)
        else:
            history[p] = np.copy(np.array(globalHistory[p]))
        if unitsInSI:
            if "unitsInSI" in globalHistory[p].attrs.keys():
                unit = globalHistory[p].attrs["unitsInSI"]
                history[p] = history[p]*unit
    return history.view(np.recarray)
