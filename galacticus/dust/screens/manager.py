#! /usr/bin/env python

class ScreenLaw(object):
    laws = {}

    @classmethod
    def register_subclass(cls, propertyType):
        def decorator(subclass):
            cls.laws[propertyType] = subclass()
            return subclass
        return decorator

