# fetal_ultrasound_project/config/config.py

import os

# --- Paths ---
# BASE_DIR should be the root of your project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'Images')
DATA_CSV_PATH = os.path.join(DATA_DIR, 'FETAL_PLANES_DB_data.csv')

TRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')
MODEL_WEIGHTS_PATH = os.path.join(TRAINED_MODELS_DIR, 'final_fetal_plane_classifier.pth')

# --- Model Parameters ---
NUM_CLASSES = 6
IMAGE_SIZE = (224, 224)  # Ensure this matches your model's input size

# --- Training Parameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # You might need to adjust this
NUM_EPOCHS = 10 # You might need to adjust this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Class Labels ---
# This MUST be in the exact order your model was trained on.
CLASS_LABELS = [
    'Fetal abdomen', 'Fetal brain', 'Fetal femur',
    'Fetal thorax', 'Maternal cervix', 'Other'
]
IDX_TO_CLASS = {i: label for i, label in enumerate(CLASS_LABELS)}
CLASS_TO_IDX = {label: i for i, label in enumerate(CLASS_LABELS)}

# --- Normalization values ---
# Use the same normalization values as during your model training.
# If you used ImageNet's default normalization, use these:
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
# If you used [0.5, 0.5, 0.5] for mean and std, use these:
# NORM_MEAN = [0.5, 0.5, 0.5]
# NORM_STD = [0.5, 0.5, 0.5]