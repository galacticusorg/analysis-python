#! /usr/bin/env python

import unittest
import galacticus.properties.manager
from galacticus.modules import findModuleChildren
_MODS = findModuleChildren(unittest.TestCase)

def _main():
    suite = unittest.defaultTestLoader.loadTestsFromNames(_MODS)
    unittest.TextTestRunner().run(suite)
    return  

if __name__ == '__main__':
    _main()

