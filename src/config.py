"""Central settings for the EAT Deep Segmentation project."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- folders ----
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")
TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, "training_log.csv")

# ---- training settings ----
IMG_SIZE = 256          # images are resized to IMG_SIZE x IMG_SIZE
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42
THRESHOLD = 0.5         # probability > THRESHOLD  ->  EAT pixel

# ---- train/val/test split ----
# "auto" -> patient-wise split when the file names look like  P01_012.png  (>= 5 patients,
#           several slices each), otherwise a random slice split.
# True   -> always patient-wise (slices of one patient never appear in both train and test).
# False  -> always random slice split.
PATIENT_SPLIT = "auto"
PATIENT_REGEX = r"^(.+?)_\d+$"

# ---- quantification settings (EDIT these for your real dataset) ----
ORIGINAL_IMAGE_SIZE = 512   # size of the original CT slice (pixels)
PIXEL_SPACING_MM = 0.7      # in-plane pixel spacing of original CT (mm)
SLICE_THICKNESS_MM = 2.5    # distance between CT slices (mm)
# prepare_nrrd_dataset.py writes data/dataset_info.json with the real values -> used automatically
_INFO = os.path.join(BASE_DIR, "data", "dataset_info.json")
if os.path.exists(_INFO):
    try:
        import json
        with open(_INFO) as _f:
            _d = json.load(_f)
        PIXEL_SPACING_MM = float(_d.get("pixel_spacing_mm", PIXEL_SPACING_MM))
        SLICE_THICKNESS_MM = float(_d.get("slice_thickness_mm", SLICE_THICKNESS_MM))
        ORIGINAL_IMAGE_SIZE = int(_d.get("original_image_size", ORIGINAL_IMAGE_SIZE))
    except Exception:
        pass
# because images are resized, the spacing per pixel grows:
EFFECTIVE_SPACING_MM = PIXEL_SPACING_MM * ORIGINAL_IMAGE_SIZE / IMG_SIZE


def ensure_dirs():
    for d in (IMAGE_DIR, MASK_DIR, MODEL_DIR, OUTPUT_DIR, PRED_DIR, GRAPH_DIR):
        os.makedirs(d, exist_ok=True)
