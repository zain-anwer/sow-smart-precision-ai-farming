#!/bin/bash

# creating virtual environment
python -m venv venv

# activating/setting virtual environment for package downloads
source venv/Scripts/activate

# upgrading the package manager pip
python -m pip install --upgrade pip

# downloading the packages mentioned in the requirements.txt
pip install -r requirements.txt
