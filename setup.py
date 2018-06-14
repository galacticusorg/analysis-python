#! /usr/bin/env python

from setuptools import setup, find_packages

setup(name='galacticus',
      version='0.1',
      description='Analysis tools for the Galacticus semi-annalytical model',
      url='https://bitbucket.org/galacticusdev/analysis-python',
      author='Alex Merson',
      author_email='alex.i.merson@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_dir={'galacticus':'galacticus'},
      zip_safe=False)

