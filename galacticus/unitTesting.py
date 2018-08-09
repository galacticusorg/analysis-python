#! /usr/bin/env python

import os,sys,fnmatch
import unittest
import pkgutil


def findUnitTests():
    unit_tests_locs = []
    pkg_dir = os.path.dirname(__file__)
    for module_loader, modname, ispkg in pkgutil.walk_packages(path=pkg_dir, onerror=lambda x: None):
        if not modname.startswith("galacticus"):
            continue
        exec('import ' + modname) in globals()
        pkg_name = modname
        obj = sys.modules[pkg_name]
        for dir_name in dir(obj):
            dir_obj = getattr(obj, dir_name)
            try:
                if issubclass(dir_obj, unittest.TestCase):
                    unit_tests_locs.append(dir_obj.__module__)
            except TypeError:
                continue
    return unit_tests_locs
