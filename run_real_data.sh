#!/bin/bash
# Run AFTER you have put your real images/masks in data/images and data/masks
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/predict.py --image data/images/P01_023.png
