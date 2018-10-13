#! /usr/bin/env python

import numpy as np

EXCEMPT = ["/parameters/mergerTreeConstructorMethod/fileNames"]

class ParametersMatch(object):
    
    @classmethod
    def matchParameter(cls,param1,param2):
        if np.ndim(param1)!=np.ndim(param2):
            return False
        if np.ndim(param1)==0:
            return param1==param2
        if len(param1) != len(param2):
            return False            
        match = all([cls.matchParameter(a,b) for a,b in zip(param1,param2)])
        return match

    @classmethod
    def match(cls,PARAMS1,PARAMS2):
        matching = list(set(PARAMS1.map).intersection(PARAMS2.map))
        match = len(matching)==len(PARAMS1.map) and len(matching)==len(PARAMS2.map)        
        if not match:
            return match
        for path in PARAMS1.map:
            if path not in EXCEMPT:
                value1 = PARAMS1.getParameter(path)
                value2 = PARAMS2.getParameter(path)
                match = match and cls.matchParameter(value1,value2)
        return match


    
