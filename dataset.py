# fault_detection_pipeline/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

# --- Data Transformations ---
# Define transformations for training (with augmentation) and validation/testing (without augmentation)
# Normalization values are standard for models pre-trained on ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Transformations for training data
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(config.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    normalize,
])

# Transformations for validation and testing data (no augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize(256), # Resize larger first
    transforms.CenterCrop(config.IMAGE_SIZE), # Crop center
    transforms.ToTensor(),
    normalize,
])

# --- Datasets ---
def create_datasets():
    """Creates training and validation datasets."""
    print(f"Loading training data from: {config.TRAIN_DIR}")
    train_dataset = datasets.ImageFolder(
        config.TRAIN_DIR,
        transform=train_transforms
    )

    print(f"Loading validation data from: {config.VAL_DIR}")
    val_dataset = datasets.ImageFolder(
        config.VAL_DIR,
        transform=val_test_transforms
    )

    # Verify classes match config
    if train_dataset.classes != config.CLASSES or val_dataset.classes != config.CLASSES:
        print("Warning: Class mismatch between datasets and config.py!")
        print(f"Dataset classes found: {train_dataset.classes}")
        print(f"Configured classes: {config.CLASSES}")
        # Consider raising an error here depending on desired strictness
        # raise ValueError("Class mismatch detected between data folders and config.py")

    return train_dataset, val_dataset

# --- Data Loaders ---
def create_dataloaders(train_dataset, val_dataset, batch_size):
    """Creates training and validation data loaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # Adjust based on your system's capabilities
        pin_memory=True # Speeds up data transfer to GPU if available
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader

if __name__ == '__main__':
    # Example usage: Load datasets and create dataloaders
    print("Testing dataset loading...")
    try:
        train_ds, val_ds = create_datasets()
        print(f"Training dataset size: {len(train_ds)}")
        print(f"Validation dataset size: {len(val_ds)}")
        print(f"Classes found: {train_ds.classes}")

        train_dl, val_dl = create_dataloaders(train_ds, val_ds, config.BATCH_SIZE)
        print(f"Number of training batches: {len(train_dl)}")
        print(f"Number of validation batches: {len(val_dl)}")

        # Fetch one batch to check
        images, labels = next(iter(train_dl))
        print(f"Batch shape: Images-{images.shape}, Labels-{labels.shape}")
        print("Dataset loading test successful!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the DATA_DIR in config.py points to the correct location")
        print("and that the 'train' and 'val' subdirectories exist with class folders.")
    except Exception as e:
        print(f"An unexpected error occurred during dataset loading: {e}")