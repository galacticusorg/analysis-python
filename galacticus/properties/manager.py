#! /usr/bin/env python

class Property(object):
    subclasses = {}

    @classmethod
    def register_subclass(cls, propertyType):
        def decorator(subclass):
            cls.subclasses[propertyType] = subclass
            return subclass
        return decorator

