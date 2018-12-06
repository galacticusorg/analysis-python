#! /usr/bin/env python

import os
from galacticus import rcParams
from galacticus.data import GalacticusData
from galacticus.utils.downloads import DownloadFromGoogleDrive


DATA = GalacticusData()
compendiumFile = rcParams.get("dustCompendium","attenuationsFile",
                              fallback="compendiumAttenuations.hdf5")
print("Downloading dust compendium file '"+compendiumFile+"' from remote location...")
file_id = "1f7UekQLCkmznZ5MppQRypWzgHscBW3S4"
destination = DATA.dynamic+"/dust/"+compendiumFile
DownloadFromGoogleDrive.download(file_id,destination)
if not os.path.exists(destination):
    raise RuntimeError("Download failed. Attempt manual download.")
print("Download complete.")
