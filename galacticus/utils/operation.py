#! /usr/bin/env python

import sys,os,subprocess,shutil
import warnings
from .. import rcParams

GALACTICUS_EXEC_PATH = rcParams.get("paths","GALACTICUS_EXEC_PATH",fallback=None)

def compileGalacticus(mpi=True,nproc=1,verbose=True,errorOnFailure=True):
    if verbose:
        print("Compiling Galacticus...")
        print("GALACTICUS SOURCE DIRECTORY = "+GALACTICUS_EXEC_PATH)
        sys.stdout.flush()
    os.chdir(GALACTICUS_EXEC_PATH)
    pwd = subprocess.check_output(["pwd"]).replace("\n","")
    assert(pwd,GALACTICUS_EXEC_PATH)
    shutil.rmtree("work/build")
    if mpi:
        if verbose:
            print("Building with MPI...")
            sys.stdout.flush()
        cmd = "make -k -j"+str(nproc)+" GALACTICUS_BUILD_OPTION=MPI Galacticus.exe"
    else:
        cmd = "make -k -j"+str(nproc)+" Galacticus.exe"
    if verbose:
        print("Executing command: '"+cmd+"'")
        sys.stdout.flush()
    os.system(cmd)
    if not os.path.exists("Galacticus.exe"):
        if errorOnFailure:
            msg = "Compilation UNSUCCESSFUL! No Galacticus executable found!"
            raise FileNotFoundError(msg)
        else:
            warnings.warn(msg)
            sys.stdout.flush()
    else:
        print("COMPILATION COMPLETED SUCCESSFULLY!")
    sys.stdout.flush()
    return


def runGalacticus(workdir,paramfile,nproc=1,exe="Galacticus.exe",mpi=True,verbose=True):    
    if not os.path.exists(workdir):
        msg = "Galacticus NOT RUN. Specified work directory '"+workdir+"' not found!"
        raise FileNotFoundError(msg)
    os.chdir(workdir)
    pwd = subprocess.check_output(["pwd"]).replace("\n","")
    assert(pwd,workdir)
    if not os.path.exists(exe):
        msg = "Galacticus NOT RUN. Galacticus executable '"+exe+"' not found!"
        raise FileNotFoundError(msg)
    if not os.path.exists(paramfile):
        msg = "Galacticus NOT RUN. Parameter file '"+paramfile+"' not found!"
        raise FileNotFoundError(msg)
    cmd = exe+" "+paramfile    
    if mpi:
        cmd = "mpirun -np "+str(nproc)+" "+cmd
    if verbose:
        print("Running command '"+cmd+"'...")
        print("Running "+exe+" using "+str(nproc)+" CPUs...")        
        sys.stdout.flush()
    os.system(cmd)
    sys.stdout.flush()
    return

    

