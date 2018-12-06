#! /usr/bin/env python

"""
Usage: mergeGalacticusOutputs.py --output <mergedfile> --search </path/to/outputs_*.hdf5> --input <file1,file2,file3> --force

"""
import sys,os,glob,fnmatch
from galacticus.fileFormats.hdf5 import HDF5
from galacticus.utils.merge import MergeGalacticusHDF5
from galacticus.utils.progress import Progress

if len(sys.argv) == 1:
    print(__doc__)
    quit()

mergeFile = None
force = False
search = None
inputfiles = None
iarg = 0
while iarg < len(sys.argv):
    if sys.argv[iarg] == "--output":
        iarg +=1 
        mergeFile = sys.argv[iarg]
    if sys.argv[iarg] == "--search":
        iarg += 1
        search = sys.argv[iarg]
    if sys.argv[iarg] == "--input":
        iarg += 1
        inputfiles = sys.argv[iarg].split(",")
    if sys.argv[iarg] == "--force":
        force = True
    iarg += 1

if inputfiles is None:
    if search is not None:
        inputfiles = glob.glob(search)
    else:
        msg = "No input files specified! Use either "+\
            "--input <file1,file2,file3,...> or --search <filePrefix.*.hdf5>"
        raise ValueError(msg)
if mergeFile is None:
    raise ValueError("No output file specified! Use --output <filename>.")



MERGE = MergeGalacticusHDF5(mergeFile)
[MERGE.appendFile(ifile,force=force) for ifile in inputfiles]


