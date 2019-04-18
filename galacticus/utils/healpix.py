#! /usr/bin/env python

import numpy as np
try: 
    import healpy as hp
except ImportError:
    raise ImportError("Unable to import Healpy!")


class Pixels(object):
    
    @classmethod
    def getPixelIndices(cls,NSIDE,ra,dec,nest=False):
        pixels = hp.ang2pix(NSIDE,dec,ra,nest=nest,lonlat=True)
        return pixels

    @classmethod
    def pixelIndexValid(cls,NSIDE,pixelIndex):
        return int(pixelIndex)>=0 and int(pixelIndex)<hp.nside2npix(NSIDE)

    @classmethod
    def getGalaxiesPixelMask(cls,NSIDE,ra,dec,pixelIndex,nest=False):        
        pixels = self.getPixelNumbersIndices(NSIDE,ra,dec,nest=nest)
        return pixels==pixelNumber
    
    
    
