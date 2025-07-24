#! /bin/bash

set -Eeuo pipefail

cd lib
python setup.py build -j 4
python -m pip install .
cd ../
pip install --disable-pip-version-check -U -r requirements.txt
python -m pip install -e .
