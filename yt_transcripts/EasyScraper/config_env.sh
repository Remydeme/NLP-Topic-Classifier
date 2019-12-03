#!/bin/bash



# configure virtual env

VIRTUAL_ENV="venv"
PYTHON_BIN_PATH="/usr/local/Cellar/python/3.7.5/bin/python3.7"

if [ -f $VIRTUAL_ENV ]
then
    rm -rv  $VIRTUAL_ENV
fi

virtualenv venv

# set virtual env ptyhon path

virtualenv -p $PYTHON_BIN_PATH

# activate the virtual env

source "$VIRTUAL_ENV/bin/activate"
