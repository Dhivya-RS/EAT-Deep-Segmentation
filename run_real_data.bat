@echo off
REM Run AFTER you have put your real images/masks in data\images and data\masks
REM (or ran:  python src\prepare_dataset.py --src "<downloaded folder>" --n 100 --clear)
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/predict.py --image data/images/P01_023.png
pause
