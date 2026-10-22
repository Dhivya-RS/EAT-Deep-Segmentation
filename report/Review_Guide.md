# Review guide - EAT Deep Segmentation

## Before the review (do this the day before)
1. Run the full pipeline once: prepare data -> `train.py` -> `evaluate.py` -> `predict.py`.
2. Check these exist: `models/best_model.h5`, `outputs/graphs/*.png`, `outputs/results.csv`,
   `outputs/predictions/*.png`.
3. Copy the whole folder to a pendrive and upload a ZIP to Google Drive (backup).
4. Colab users: run the notebook top to bottom, then File -> Download -> .ipynb, and also
   File -> Print -> Save as PDF (keeps all outputs).

## Order to show in the review
1. Problem: what is EAT, why measure it (cardiac risk)
2. Dataset: Siriapisith et al. 2021 (PeerJ CS 7:e806), figshare DOI 10.6084/m9.figshare.17039813; CT volumes with
   radiologist EAT ground truth; how many patients/slices you used (see data/pairs.csv); patient-wise split
3. Method: pre-processing, U-Net, BCE + Dice loss
4. Live demo: `python src/predict.py --image data/images/P01_023.png`
5. Results: Dice, IoU, precision, recall (`outputs/results.csv`), training curves
6. Quantification: EAT area (cm2) / volume (cm3) and how it is computed
7. Folder structure: open the folder in Explorer / VS Code
8. Limitations: small dataset, 2D slices only, no external validation

## Possible viva questions
* What is epicardial adipose tissue? - fat between the myocardium and the visceral pericardium.
* Why U-Net? - encoder-decoder with skip connections; works well with small medical datasets.
* What are Dice and IoU? - overlap measures between predicted and true mask (1 = perfect).
* Why BCE + Dice loss? - EAT pixels are a tiny part of the image (class imbalance); Dice handles that.
* How did you avoid overfitting? - validation set, early stopping, LR reduction, dropout, augmentation.
* How is area / volume computed? - EAT pixels x (pixel spacing)^2 ; volume = sum of slice areas x slice thickness.
* How many images? Split? - say the real numbers from `data/pairs.csv` / the train.py printout.
* Is this clinically usable? - No, it is a research prototype; needs larger data and clinical validation.
