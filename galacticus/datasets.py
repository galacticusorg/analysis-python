#! /usr/bin/env python

class Dataset(object):

    def __init__(self,name=None,data=None,attr={}):
        self.name = name
        self.data = data
        self.attr = attr
        return

    def reset(self):
        self.name = None
        self.data = None
        self.attr = {}
        return

    
