#!/bin/sh
# Parsing svmlight based numerical feature vectors

BASE_DIR="/ivi/ilps/datasets/istella22"

python main.py parse-vectors $BASE_DIR train
python main.py parse-vectors $BASE_DIR valid
python main.py parse-vectors $BASE_DIR test
