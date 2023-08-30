#! /usr/bin/env python

import os,sys
import warnings
import numpy as np
from ..fileFormats.hdf5 import HDF5
from ..parameters.io import ParametersFromHDF5,ParametersToHDF5
from ..parameters.compare import ParametersMatch
from .progress import Progress
from .timing import stopwatch

class MergeGalacticusHDF5(object):

    def __init__(self,outfile):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.OUT = HDF5(outfile,'w')
        return

    def delete(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        path = self.OUT.filename
        self.OUT.close()
        os.remove(path)
        return

    def updateUUID(self,hdfObj):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print(funcname+"(): Updating UUID")
        uuid = ""
        if "UUID" in self.OUT.readAttributes("/"):
            uuid = self.OUT.readAttributes("/",required=["UUID"])["UUID"]+":"
        uuid = uuid + hdfObj.readAttributes("/",required=["UUID"])["UUID"]
        self.OUT.addAttributes("/",{"UUID":uuid},overwrite=True)
        return
    
    def checkAttributes(self,hdfObj,path,exempt=[],forceMerge=False):
        attr = hdfObj.readAttributes(path)
        outAttr = self.OUT.readAttributes(path)
        check = all([outAttr[key]==attr[key] for key in attr.keys() if key not in list(exempt)])
        if not check:
            if not forceMerge:
                self.delete()
                raise ValueError(funcname+"(): attributes are not consistent!")
        return

    def updateVersion(self,hdfObj,forceMerge=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print(funcname+"(): Updating version information")
        attr = hdfObj.readAttributes("/Version")
        if "Version" not in self.OUT.lsGroups("/"):
            self.OUT.cpGroup(hdfObj.filename,"/Version")
            return 
        self.checkAttributes(hdfObj,"/Version",exempt=["runTime"],forceMerge=forceMerge)
        return 

    def updateBuild(self,hdfObj,forceMerge=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print(funcname+"(): Updating build information")
        attr = hdfObj.readAttributes("/Build")
        if "Build" not in self.OUT.lsGroups("/"):
            self.OUT.cpGroup(hdfObj.filename,"/Build")
            return 
        exempt = ["make_PREPROCESSOR","make_MODULETYPE","make_FCFLAGS_NOOPT",
                  "make_FCFLAGS","make_FCCOMPILER_VERSION","make_FCCOMPILER",
                  "make_CPPCOMPILER_VERSION","make_CPPCOMPILER",
                  "make_CCOMPILER_VERSION","make_CCOMPILER"]
        self.checkAttributes(hdfObj,"/Build",exempt=exempt,forceMerge=forceMerge)
        return 
            
    def fileComplete(self,hdfObj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        name = "statusCompletion"
        attr = hdfObj.readAttributes("/")
        if name not in attr.keys():
            return False
        completed = int(attr[name])
        return bool(completed)

    def updateDataset(self,hdfObj,path,PROG=None):
        name = path.split("/")[-1]        
        parent = path.replace("/"+name,"")
        data = hdfObj.readDataset(path,exit_if_missing=True)
        if name in self.OUT.lsDatasets(parent):
            self.OUT.addDataset(parent,name,data,append=True)
        else:
            self.OUT.addDataset(parent,name,data,append=False)
            attr = hdfObj.readAttributes(path)
            self.OUT.addAttributes(path,attr)
        if PROG is not None:
            PROG.increment()
            PROG.print_status_line()
        return
                
    def updateMetaData(self,hdfObj):
        if "metaData" not in hdfObj.lsGroups("/"):
            return
        if "metaData" not in self.OUT.lsGroups("/"):
            self.OUT.cpGroup(hdfObj.filename,"/metaData")
        path = "/metaData/treeTiming"
        [self.updateDataset(hdfObj,path+"/"+dset) for dset in hdfObj.lsDatasets(path)]
        return
        
    def updateMergerTreeData(self,hdfObj,output):
        print("   ---> Updating merger tree data")
        path = "/Outputs/"+output
        dsets = hdfObj.lsDatasets(path)
        [self.updateDataset(hdfObj,path+"/"+dset) for dset in dsets if dset is not "nodeData"]
        return
    
    def updateNodeData(self,hdfObj,output):
        print("   ---> Updating node data")
        path = "/Outputs/"+output+"/nodeData"
        if "nodeData" not in self.OUT.lsGroups("/Outputs/"+output):
            self.OUT.mkGroup(path)
            attr = hdfObj.readAttributes(path)
            self.OUT.addAttributes(path,attr)
        dsets = hdfObj.lsDatasets(path)
        PROG = Progress(len(dsets))
        [self.updateDataset(hdfObj,path+"/"+dset,PROG=PROG) for dset in dsets]
        return
            
    def updateSingleOutput(self,hdfObj,output):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if "Outputs" not in self.OUT.lsGroups("/"):
            self.OUT.mkGroup("/Outputs")
        path = "/Outputs/"+output
        print("   Updating "+path)
        if output not in self.OUT.lsGroups("/Outputs"):
            self.OUT.mkGroup(path)
            attr = hdfObj.readAttributes(path)
            self.OUT.addAttributes(path,attr)
        else:
            self.checkAttributes(hdfObj,path)
        self.updateMergerTreeData(hdfObj,output)
        self.updateNodeData(hdfObj,output)
        return
    
    def updateOutputs(self,hdfObj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print(funcname+"(): Updating Outputs")
        if "Outputs" not in hdfObj.lsGroups("/"):
            return
        outputs = hdfObj.lsGroups("/Outputs")
        if len(outputs) == 0:
            return
        [self.updateSingleOutput(hdfObj,str(output)) for output in outputs]
        return

    def updateParameters(self,hdfObj,force=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print(funcname+"(): Updating parameters")
        PARAMS = ParametersFromHDF5.read(hdfObj)        
        if "Parameters" not in self.OUT.lsGroups("/"):
            self.OUT.cpGroup(hdfObj.filename,"/Parameters")
        else:
            OUT_PARAMS = ParametersFromHDF5.read(self.OUT)
            if not ParametersMatch.match(PARAMS,OUT_PARAMS,force=force):                
                raise RuntimeError(funcname+"(): Parameters in file '"+hdfObj.filename+"' do not match.")
            exempt = ParametersMatch.exempt(PARAMS,OUT_PARAMS)
            if len(exempt)>0:
                ParametersToHDF5.write(self.OUT,PARAMS,append=True)
        return

    def updateGlobalHistoryDataset(self,name,data):
        DATA = self.OUT.fileObj["/globalHistory/"+name]
        DATA[:] += data
        return

    def updateGlobalHistory(self,hdfObj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print(funcname+"(): Updating global history")
        if "globalHistory" not in hdfObj.lsGroups("/"):
            return
        if "globalHistory" not in self.OUT.lsGroups("/"):
            self.OUT.cpGroup(hdfObj.filename,"globalHistory")
            return
        datasets = self.OUT.lsDatasets("/globalHistory")
        datasets = list(set(datasets).difference(["historyTime","historyExpansion"]))
        [self.updateGlobalHistoryDataset(dset,hdfObj.readDataset("/globalHistory/"+dset)) 
         for dset in datasets]
        return

    def appendFile(self,fname,force=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("Merging file '"+fname+"'...")
        STOP = stopwatch()
        HDF = HDF5(fname,'r')
        if not self.fileComplete(HDF):
            if not force:
                self.delete()
                raise RuntimeError(funcname+"(): file "+fname+" is corrupted or not complete!")
            else:
                warnings.warn(funcname+"(): file "+fname+" may be corrupted or not complete! Forcing merge.")
        self.updateUUID(HDF)
        self.updateVersion(HDF)
        self.updateBuild(HDF)       
        self.updateParameters(HDF,force=force)
        self.updateOutputs(HDF)
        self.updateGlobalHistory(HDF)
        HDF.close()
        print("Merging complete.")
        STOP.stop()
        return
    

        
