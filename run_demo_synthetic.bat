@echo off
REM DEMO ONLY with synthetic data (not real CT).
pip install -r requirements.txt
python src/generate_sample_data.py
python src/train.py
python src/evaluate.py
python src/predict.py --image data/images/sample_0001.png
pause
