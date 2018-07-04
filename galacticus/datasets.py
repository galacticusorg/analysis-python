#! /usr/bin/env python

class Dataset(object):

    def __init__(self,name=None,data=None,path=None,unitsInSI=None):
        self.name = name
        self.data = data
        self.path = path
        self.unitsInSI = unitsInSI
        return

    def reset(self):
        self.name = None
        self.data = None
        self.path = None
        self.unitsInSI = None
        return

    
