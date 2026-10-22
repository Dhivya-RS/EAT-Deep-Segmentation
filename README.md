# EAT Deep Segmentation
Deep Learning-Based Automated Segmentation and Quantification of Epicardial Adipose Tissue (EAT) from Cardiac CT Images.

A U-Net segments the epicardial fat on a cardiac CT slice. The predicted mask is then used to
quantify EAT (area per slice, volume for a slice stack).

## Folder structure
```
EAT_Deep_Segmentation/
├── data/images/     CT slices (png/jpg/tif)
├── data/masks/      ground-truth EAT masks (same file names as images)
├── models/          best_model.h5  (created by train.py)
├── src/
│   ├── config.py                 settings (paths, image size, pixel spacing ...)
│   ├── generate_sample_data.py   synthetic demo data (only if you have no real dataset)
│   ├── prepare_nrrd_dataset.py   NRRD CT volumes + labels -> PNG slices (figshare EAT dataset)
│   ├── nrrd_io.py                NRRD reader
│   ├── prepare_dataset.py        PNG/JPG image+mask folders -> data/images + data/masks
│   ├── data_loader.py            load / resize / pair / 70-15-15 split (random or patient-wise)
│   ├── model.py                  U-Net + Dice/IoU + BCE-Dice loss
│   ├── train.py                  training + curves
│   ├── evaluate.py               test metrics -> outputs/results.csv + sample images
│   ├── predict.py                demo: image in -> mask + EAT area out
│   ├── quantify.py               area (cm2) and volume (cm3)
│   ├── metrics_np.py, visualize.py
├── outputs/
│   ├── graphs/        loss / Dice / IoU curves
│   ├── predictions/   original | ground truth | predicted | overlay
│   ├── results.csv    Dice, IoU, precision, recall, EAT area per test image
│   └── training_log.csv
├── report/            Review_Guide.md (demo order + viva Q&A); put report PDF / PPT here
├── EAT_Colab_Notebook.ipynb   run everything on Google Colab
├── requirements.txt
└── run_real_data.* / run_demo_synthetic.*
```

## How to run
1. Install Python 3.9 - 3.11 and then:  `pip install -r requirements.txt`
2. **Prepare the real dataset.** Recommended: the public EAT dataset of Siriapisith, Kusakunniran &
   Haddawy (2021), figshare DOI 10.6084/m9.figshare.17039813 (cardiac CT as NRRD volumes with
   radiologist EAT ground truth; folders Epicardial_train_NC / test_NC / train_CE / test_CE).
   Download (1.22 GB), extract, then:
   ```
   python src/prepare_nrrd_dataset.py --src "C:/path/to/dataset" --inspect
   python src/prepare_nrrd_dataset.py --src "C:/path/to/dataset" --filter train_NC --patients 12 --slices 8 --clear
   ```
   This converts volumes to PNG slices (`data/images/P01_023.png`, masks with the same names), keeps slices
   that really contain EAT, spreads them over 12 patients and stores pixel spacing / slice thickness from the
   NRRD header in `data/dataset_info.json` (used automatically by quantification).
   With patient ids in the file names the split is **patient-wise** (no slice of a test patient is in training).
   *(Other PNG/JPG datasets: `python src/prepare_dataset.py --src <folder> --n 100 --clear`.
   Demo without any dataset: `python src/generate_sample_data.py` = SYNTHETIC images, not for reporting results.)*
3. Train:     `python src/train.py`
4. Evaluate:  `python src/evaluate.py`
5. Demo:      `python src/predict.py --image data/images/P01_023.png   (use any file name from data/images)`

Shortcuts: `run_real_data.bat` / `run_real_data.sh` (steps 3-5), `run_demo_synthetic.*` (synthetic demo),
`EAT_Colab_Notebook.ipynb` (everything on Google Colab with outputs kept in the notebook).

## Important settings (src/config.py)
`PATIENT_SPLIT = "auto"` (default) uses a patient-wise split when file names look like `P01_023.png`.
`PIXEL_SPACING_MM` and `SLICE_THICKNESS_MM` decide the area / volume values. Set them to the real
values of your CT dataset (DICOM tags PixelSpacing and SliceThickness).

## Method summary
* Pre-processing: grayscale, resize to 256x256, scale to [0,1]
* Model: U-Net (4 down / 4 up blocks, skip connections, sigmoid output)
* Loss: Binary cross-entropy + Dice loss; optimizer Adam; early stopping + LR reduction
* Metrics: Dice, IoU, precision, recall, EAT area error
* Quantification: area = EAT pixels x spacing^2 ; volume = sum(area) x slice thickness
