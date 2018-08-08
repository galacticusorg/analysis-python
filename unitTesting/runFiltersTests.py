#! /usr/bin/env python

import unittest

# Edit this to include new modules or TestCases to run.                                                                                                                         
_MODS = ['galacticus.filters','galacticus.filters.filters']

def _main():
    suite = unittest.defaultTestLoader.loadTestsFromNames(_MODS)
    unittest.TextTestRunner().run(suite)
    return  

if __name__ == '__main__':
    _main()

