#! /usr/bin/env python

import sys
import numpy as np
import warnings

EXEMPT = ["/parameters/mergerTreeConstructorMethod/fileNames"]

class ParametersMatch(object):
    
    @classmethod
    def matchParameter(cls,param1,param2):
        return np.array_equal(param1,param2)

    @classmethod
    def missing(cls,PARAMS1,PARAMS2):
        missing = list(set(PARAMS1.map).difference(PARAMS2.map)) + \
            list(set(PARAMS2.map).difference(PARAMS1.map))
        return missing

    @classmethod
    def common(cls,PARAMS1,PARAMS2):
        return list(set(PARAMS1.map).intersection(PARAMS2.map))

    @classmethod
    def matching(cls,PARAMS1,PARAMS2):
        paths = cls.common(PARAMS1,PARAMS2)        
        P1 = PARAMS1.getParameter
        P2 = PARAMS2.getParameter
        matching = [path for path in paths if cls.matchParameter(P1(path),P2(path))]
        return matching

    @classmethod
    def different(cls,PARAMS1,PARAMS2):
        paths = cls.common(PARAMS1,PARAMS2)        
        P1 = PARAMS1.getParameter
        P2 = PARAMS2.getParameter
        different = [path for path in paths if not cls.matchParameter(P1(path),P2(path))]
        return different

    @classmethod
    def exempt(cls,PARAMS1,PARAMS2):
        different = cls.different(PARAMS1,PARAMS2)
        return list(set(different).intersection(EXEMPT))
                    
    @classmethod
    def match(cls,PARAMS1,PARAMS2,force=False):
        funcname = cls.__class__.__name__+"."+sys._getframe().f_code.co_name
        missing = cls.missing(PARAMS1,PARAMS2)
        if len(missing)>0:
            if force:
                msg = funcname+"(): Some parameters are missing:"+",".join(missing)+"."
                warnings.warn(msg)
            else:
                return False
        different = cls.different(PARAMS1,PARAMS2)        
        if len(list(set(different).difference(EXEMPT)))>0:
            return False
        return True

    
