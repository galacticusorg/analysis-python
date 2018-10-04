#! /usr/bin/env python

import sys,os,glob,fnmatch
import numpy as np
from ..filters.filters import GalacticusFilter
from . import GalacticusParameters

allowedPostProcessing = ["default","recent","unabsorbed","recentUnabsorbed"]
allowedAbsorptionMethods = ["inoue2014","meiksin2006","madau1995","lycSuppress","identity"]

class FilterParameterSet(object):
        
    def updateParameterList(self,PARAMS,listName,paramName):    
        paramlist = PARAMS.getParameter(listName)
        if paramlist is None:
            paramlist = paramName
        else:
            paramlist = paramlist + " " + paramName
        PARAMS.setParameter(listName,paramlist)
        return

    def updateMethod(self,PARAMS,methodPath,method):
        methods = PARAMS.getParameter(methodPath)
        if methods is None:
            methods = []
        else:
            methods = methods.split()
        if method not in methods:
            methods.append(method)
            methods = " ".join(methods)
            PARAMS.setParameter(methodPath,methods)
        return

    def updateRecentMethod(self,PARAMS,methodPath,method):
        methods = PARAMS.getParameter(methodPath)
        if methods is None:
            methods = []
        else:
            methods = methods.split()
        if method not in methods or "recent" not in methods:
            methods = methods + [method,"recent"]
            methods = list(np.unique(methods))
            PARAMS.setParameter(methodPath,methods)
        return

    def addFilterSet(self,PARAMS,filterName,frame,redshift="all",postProcess="default",absorption="inoue2014"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check inputs
        if frame not in ["rest","observed"]:
            raise ValueError(funcname+"(): frame '"+postProcess+"' not recognized. Allowed "+
                             "frames are 'rest' or 'observed'.")    
        if postProcess not in allowedPostProcessing:
            raise ValueError(funcname+"(): postprocessing method '"+postProcess+"' not recognized. Allowed "+
                             "methods include: "+",".join(allowedPostProcessing)+".")
        if absorption not in allowedAbsorptionMethods:
            raise ValueError(funcname+"(): absorption method '"+absorption+"' not recognized. Allowed "+
                             "methods include: "+",".join(allowedAbsorptionMethods)+".")
        # Update filter names
        self.updateParameterList(PARAMS,"/parameters/luminosityFilter",filterName)
        # Update frames
        self.updateParameterList(PARAMS,"/parameters/luminosityType",frame)
        # Update redshift
        self.updateParameterList(PARAMS,"/parameters/luminosityRedshift",str(redshift))
        # Update post-processing method
        self.updateParameterList(PARAMS,"/parameters/luminosityPostprocessSet",postProcess)
        # Update methods
        self.updateMethod(PARAMS,"/parameters/stellarPopulationSpectraPostprocessDefaultMethods",absorption)
        if postProcess == "recent":        
            self.updateRecentMethod(PARAMS,"/parameters/stellarPopulationSpectraPostprocessRecentMethods",absorption)
        if postProcess == "unabsorbed":
            self.updateMethod(PARAMS,"/parameters/stellarPopulationSpectraPostprocessUnabsorbedMethods","identity")
        if postProcess == "recentUnabsorbed":
            self.updateRecentMethod(PARAMS,"/parameters/stellarPopulationSpectraPostprocessRecentUnabsorbedMethods","identity")        
        return 
    
        

