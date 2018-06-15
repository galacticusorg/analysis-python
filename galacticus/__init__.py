#! /usr/bin/env python

import __future__
import os
import warnings

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

    
