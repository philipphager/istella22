#!/bin/sh
# Parsing svmlight based numerical feature vectors

BASE_DIR="/ivi/ilps/datasets/istella22"

python main.py $BASE_DIR train
python main.py $BASE_DIR val
python main.py $BASE_DIR test
