#! /usr/bin/env python

import os,sys,fnmatch
import pkgutil
import unittest


def _main():
    loader = unittest.TestLoader()
    path = "/".join(__file__.split("/")[:-1])
    suite = loader.discover(path)
    runner = unittest.TextTestRunner()
    runner.run(suite)
    return

if __name__ == '__main__':
    _main()
