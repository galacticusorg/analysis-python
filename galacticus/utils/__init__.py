#! /usr/bin/env python

import __future__
import numpy as np

def match_dimensions(*args):
    """
    match_dimensions(): Check all specified arguments have same dimension.
    """    
    if len(args) == 2:
        arg0 = np.array(args[0])
        arg1 = np.array(args[1])
        same = arg0.shape==arg1.shape
    else:
        same = [match_dimensions(args[0],arg) for arg in args[1:]]
        same = all(same)
    return same


