#! /usr/bin/env python

"""
galacticus.datasets
===================

Module containing :class:`~galacticus.datasets.Dataset` class, which is used to store Galacticus node datasets.

"""

class Dataset(object):
    """
    Base class used to store Galacticus node dataset.

    Arguments:
        name (str,optional) : Name of node dataset.
        data (array_like,{N,},optional) : Numpy array of node data.
        attr (dict,optional) : Dictionary of node dataset attributes.

    Attributes:
        name (str) : Name of node dataset.
        data (array_like,{N,}) : Numpy array of node data.
        attr (dict) : Dictionary of node dataset attributes.

    """

    def __init__(self,name=None,data=None,attr={}):
        self.name = name
        self.data = data
        self.attr = attr
        return

    def reset(self):
        """
        Reset instance of :class:`~galacticus.datasets.Dataset`.
        """
        self.name = None
        self.data = None
        self.attr = {}
        return

    
