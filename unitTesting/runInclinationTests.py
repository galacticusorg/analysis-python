#! /usr/bin/env python

import unittest
import galacticus.properties.manager

# Edit this to include new modules or TestCases to run.
_MODS = ['galacticus.inclination']

def _main():
    suite = unittest.defaultTestLoader.loadTestsFromNames(_MODS)
    unittest.TextTestRunner().run(suite)
    return  

if __name__ == '__main__':
    _main()

