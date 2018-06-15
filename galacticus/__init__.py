#! /usr/bin/env python

import __future__
import os
import warnings

# Load path to Galacticus datasets
DATASETS_PATH = None
datasetsKeyName = "GALACTICUS_DATASETS"
if datasetsKeyName in os.environ.keys():
    DATASETS_PATH = os.environ[datasetsKeyName]
    if not DATASETS_PATH.endswith("/"):
        DATASETS_PATH = DATASETS_PATH + "/"
else:
    warningString = "WARNING! No path specified for Galacticus datasets.\n"+\
        "Specify the path in your environment variables using the variable name '"+\
        datasetsKeyName+"'."
    warnings.warn(warningString)

# Load path to Galacticus source code
GALACTICUS_PATH = None
datasetsKeyName = "GALACTICUS_ROOT"
if datasetsKeyName in os.environ.keys():
    GALACTICUS_PATH = os.environ[datasetsKeyName]
    if not GALACTICUS_PATH.endswith("/"):
        GALACTICUS_PATH = GALACTICUS_PATH + "/"
else:
    warningString = "WARNING! No path specified for Galacticus source code.\n"+\
        "Specify the path in your environment variables using the variable name '"+\
        datasetsKeyName+"'."
    warnings.warn(warningString)


def checkDatasetsPath():
    if DATASETS_PATH is None:
        msg = "ERROR! Unable to locate Galacticus datasets. "+\
            "DATASETS_PATH = "+str(DATASETS_PATH)
        raise RuntimeError(msg)
    if not os.path.exists(DATASETS_PATH):
        msg = "ERROR! Datasets path '"+DATASETS_PATH+"' does not exist."
        raise RuntimeError(msg)
    if not os.path.exists(DATASETS_PATH+"static"):
        msg = "ERROR! Static datasets path '"+DATASETS_PATH+"static' does not exist."
        raise RuntimeError(msg)
    if not os.path.exists(DATASETS_PATH+"dynamic"):
        os.makedirs(DATASETS_PATH+"dynamic")
    return
