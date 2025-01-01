#!/bin/bash

set -e

BIN=bin
PYTHON=python
PIP=pip

case "$OSTYPE" in
  darwin*)
    echo "OSX"
    PYTHON=python3
    PIP=pip3
    ;; 
  linux*)
    echo "LINUX"
    PYTHON=python3
    PIP=pip3
    ;;
  msys*)
    echo "WINDOWS"
    BIN=Scripts
    ;;
  *)
    echo "unknown: $OSTYPE"
    ;;
esac

#python3 --version

rm -rf venv
$PYTHON -m venv venv

venv/$BIN/$PYTHON -m pip install --upgrade pip 
##venv/$BIN/$PYTHON -m pip install 
###venv/$BIN/$PIP install --no-cache-dir -r requirements.txt
venv/$BIN/$PIP install -r requirements.txt
