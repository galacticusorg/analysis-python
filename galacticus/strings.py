#! /usr/bin/env python

import six
import numpy as np

def removeByteStrings(value):
    if np.ndim(value)==0:
        result = value
        if isinstance(value,bytes):
            result = str(value,encoding='utf-8')
        return result
    if isinstance(value,dict):
        result = {removeByteStrings(key):removeByteStrings(value[key]) for key in value.keys()}
    elif isinstance(value,np.ndarray):
        result = [removeByteStrings(x) for x in value]
        result = np.array(result)
    else:
        dt = type(value)
        result = [removeByteStrings(x) for x in value]
        result = dt(result)
    return result

def addByteStrings(value):
    if np.ndim(value)==0:
        result = value
        if isinstance(result,str):
            result = np.string_(result)
        return result
    if isinstance(value,dict):
        result = {addByteStrings(key):addByteStrings(value[key]) for key in value.keys()}
    elif isinstance(value,np.ndarray):
        result = [addByteStrings(x) for x in value]
        result = np.array(result)
    else:
        dt = type(value)
        result = [addByteStrings(x) for x in value]
        result = dt(result)
    return result
