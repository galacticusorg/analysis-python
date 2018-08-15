#! /usr/bin/env python

import numpy as np
from ...constants import angstrom,micron

def colorRatio(wavelength,galaxy):
    funcname = sys._getframe().f_code.co_name    
    # Colour ratio parameters
    galaxies = {}
    galaxies["MW"] = {"invLambda0":4.595,"gamma":1.051,"C1":-0.38,"C2":0.74,"C3":3.96,"C4":0.26}
    galaxies["LMC"] = {"invLambda0":4.608,"gamma":0.994,"C1":-0.69,"C2":0.89,"C3":2.55,"C4":0.50}
    # Store array of inverted wavelength
    invLambda = 1.0/wavelength
    # Extract arrays of colour ratio parameters
    if galaxy.upper() not in ["MW","LMC"]:
        raise ValueError(funcname+"(): Galaxy '"+galaxy+"' not recognized. Should be 'MW' or 'LMC'.")    
    params = galaxies[galaxy]
    invLambda0 = params["invLambda0"]*np.ones_like(wavelength)
    gamma = params["gamma"]*np.ones_like(wavelength)
    C1 = params["C1"]*np.ones_like(wavelength)
    C2 = params["C2"]*np.ones_like(wavelength)
    C3 = params["C3"]*np.ones_like(wavelength)
    C4 = params["C4"]*np.ones_like(wavelength)
    mask = invLambda < 5.9
    np.place(C4,mask,0.0)
    # Compute colour ratio
    factor2 = C2*invLambda
    factor3 = C3/((invLambda-(invLambda0**2/invLambda))**2+gamma**2)
    factor4 = C4*(0.539*(invLambda-5.9)**2+0.0564*(invLambda-5.9)**3)
    ratio = C1+factor2+factor3+factor4
    return ratio

def getAllenDustTable():
    table = np.zeros(20,dtype=[("wavelength",float),("klambda",float)]).view(np.recarray)
    table.wavelength = np.array([1000., 1110., 1250., 1430., 1670., 2000., 2220., 2500., \
                                     2850., 3330., 3650., 4000., 4400., 5000., 5530., 6700., \
                                     9000., 10000., 20000., 100000.])
    table.wavelength *= angstrom/micron
    table.klambda = np.array([4.20, 3.70, 3.30, 3.00, 2.70, 2.80, 2.90, 2.30, 1.97, 1.69,\
                                  1.58, 1.45, 1.32, 1.13, 1.00, 0.74, 0.46, 0.38, 0.11,0.00])
    return table
    
