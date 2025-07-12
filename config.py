# fault_detection_pipeline/config.py
import torch
import os

# --- Dataset Configuration ---
# Adjust DATA_DIR to the absolute path of your 'data' folder
# Example: DATA_DIR = '/home/user/projects/fault_detection_pipeline/data'
# Or use os.path.abspath for relative paths
BASE_DIR = "F:/3dprinting"
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'val')

# --- Model Saving ---
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Create dir if it doesn't exist

# --- Class Labels ---
# Make sure this list exactly matches your folder names in train/val/test
CLASSES = [
    'Class 1 - Spaghetti',
    'Class 2 - Stringing and oozing',
    'Class 3 - Blobs and Zits',
    'Class 4 - Warping',
    'Class 5 - Z Seam',
    'Class 6 - Layer Shifting',
    'Class 7 - Layer Separation',
    'Class 8 - Overhang Defects',
    'Class 9 - Okay',
]
NUM_CLASSES = len(CLASSES)

# --- Models to Train/Use ---
# Choose models from torchvision.models or define custom ones
# Examples: 'resnet18', 'resnet34', 'vgg16', 'mobilenet_v2', 'efficientnet_b0'
MODEL_NAMES = [
    'resnet18',          # Existing
    'resnet34',          # Existing
    'mobilenet_v2',      # Existing
    'efficientnet_b0',   # Existing
    'resnet50',          # New
    'densenet121',       # New
    'efficientnet_b4',   # New
    'vit_b_16',          # New (Vision Transformer)
    'swin_t',            # New (Swin Transformer)
    'vgg16'            # Existing
]# Start with 4 models

# --- Training Hyperparameters ---
NUM_EPOCHS = 120  # Adjust as needed
BATCH_SIZE = 80  # Adjust based on GPU memory
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224 # Standard size for many pre-trained models
NUM_WORKERS =4
# --- Computing Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- GUI Configuration ---
DEFAULT_IMAGE_PATH = None # Or path to a default placeholder image for the GUI

USE_PRETRAINED = True        # Whether to use pretrained weights for models
WEIGHT_DECAY = 1e-4          # Weight decay for optimizer
SCHEDULER_STEP_SIZE = 10      # StepLR: step size
SCHEDULER_GAMMA = 0.2 