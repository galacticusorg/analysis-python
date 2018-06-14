#! /usr/bin/env python

import __future__
import os
import warnings

# Load path to Galacticus datasets
DATASETS_PATH = None
datasetsKeyName = "GALACTICUS_DATASETS"
if datasetsKeyName in os.environ.keys():
    DATASETS_PATH = os.environ[datasetsKeyName]
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
else:
    warningString = "WARNING! No path specified for Galacticus source code.\n"+\
        "Specify the path in your environment variables using the variable name '"+\
        datasetsKeyName+"'."
    warnings.warn(warningString)

