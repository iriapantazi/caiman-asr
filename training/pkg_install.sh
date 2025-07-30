#! /bin/bash

set -Eeuo pipefail

cd lib
python setup.py build -j 4
python -m pip install .
cd ../
pip install --disable-pip-version-check -U -r requirements.txt
python -m pip install -e .
pip install --no-dependencies torchdata==0.6.1
python ./caiman_asr_train/data/make_datasets/librispeech.py --dataset_parts dev-clean
mkdir -p /datasets/stats
mkdir -p /datasets/sentencepieces
cp open1.8-beta1023.model /datasets/sentencepieces/.
cp open1.8-beta1023.vocab /datasets/sentencepieces/.
cp -r open1.8-beta-winsz0.02 /datasets/stats/.
