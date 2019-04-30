#!/bin/bash

sphinx-apidoc -e -f -o source/ ../galacticus/

sphinx-build -b html source/ build/
