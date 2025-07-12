# fault_detection_pipeline/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
import argparse # To select which model to train (currently not used this way in main)
import json # For saving history
from tqdm import tqdm # For progress bars

import config
import dataset
import models

def train_model(model, model_name, criterion, optimizer, scheduler, dataloaders, num_epochs=25, device='cpu'):
    """
    Trains the model with enhanced logging, history saving, and model checkpointing.

    Args:
        model (torch.nn.Module): The model to train.
        model_name (str): Name of the model (for saving).
        criterion: Loss function.
        optimizer: Optimization algorithm.
        scheduler: Learning rate scheduler.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        num_epochs (int): Number of training epochs.
        device (str): 'cuda' or 'cpu'.

    Returns:
        torch.nn.Module: The model loaded with the best performing weights on validation set.
        dict: Training history (losses and accuracies).
    """
    since = time.time()

    # Ensure model save directory exists
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    print(f"Models and history will be saved in: {config.MODEL_SAVE_DIR}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    current_best_model_filepath = None # To keep track of the current best model file

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"\nStarting training for model: {model_name} for {num_epochs} epochs on {device}")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            dataset_size = len(dataloaders[phase].dataset)

            # Using tqdm for progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}", unit="batch")

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update tqdm postfix
                progress_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / inputs.size(0))

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"{phase.capitalize():<6} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'val':
                if epoch_acc > best_acc:
                    old_best_acc = best_acc
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # New filename for the best model including accuracy
                    new_best_filename = f"{model_name}_best_val_acc_{best_acc:.4f}.pth"
                    new_best_filepath = os.path.join(config.MODEL_SAVE_DIR, new_best_filename)
                    
                    torch.save(model.state_dict(), new_best_filepath)
                    print(f"🎉 New best model saved: {new_best_filename} (Val Acc: {best_acc:.4f} > {old_best_acc:.4f})")

                    # Remove the previous best model file if it exists and is different
                    if current_best_model_filepath and os.path.exists(current_best_model_filepath) and current_best_model_filepath != new_best_filepath:
                        try:
                            os.remove(current_best_model_filepath)
                            print(f"   Removed old best model: {os.path.basename(current_best_model_filepath)}")
                        except OSError as e:
                            print(f"   Error removing old best model {os.path.basename(current_best_model_filepath)}: {e}")
                    current_best_model_filepath = new_best_filepath
                else:
                    print(f"Validation accuracy ({epoch_acc:.4f}) did not improve from best ({best_acc:.4f}).")


        if phase == 'train': # Scheduler step should be after optimizer.step()
             scheduler.step() # Step the learning rate scheduler (typically after validation or training phase based on scheduler type)

        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")

    time_elapsed = time.time() - since
    print(f'\n--- Training Summary for {model_name} ---')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Accuracy: {best_acc:.4f} (achieved in file: {os.path.basename(current_best_model_filepath) if current_best_model_filepath else "N/A"})')

    # --- 1. Save the model from the very last epoch ---
    last_epoch_model_filename = f"{model_name}_epoch_{num_epochs}_last.pth"
    last_epoch_model_path = os.path.join(config.MODEL_SAVE_DIR, last_epoch_model_filename)
    torch.save(model.state_dict(), last_epoch_model_path) # Save current model state (end of last epoch)
    print(f"Model from last epoch saved to: {last_epoch_model_filename}")

    # --- 2. The best model is already saved with accuracy in its name (current_best_model_filepath) ---
    # No need to save `_final.pth` separately if current_best_model_filepath is the definitive best.
    # The 'best_model_wts' are loaded below before returning the model.

    # --- 3. Save training history ---
    history_filename = f"{model_name}_training_history.json"
    history_path = os.path.join(config.MODEL_SAVE_DIR, history_filename)
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to: {history_filename}")
    except Exception as e:
        print(f"Error saving training history: {e}")

    # Load best model weights to return the best model
    model.load_state_dict(best_model_wts)
    return model, history

def main(model_to_train):
    """Main function to setup and run training for a specific model."""
    print(f"\n=================================================")
    print(f"--- Initializing Training for Model: {model_to_train.upper()} ---")
    print(f"=================================================")

    # --- 1. Load Data ---
    print("\n--- 1. Loading Data ---")
    try:
        train_ds, val_ds = dataset.create_datasets()
        train_loader, val_loader = dataset.create_dataloaders(
            train_ds, val_ds, config.BATCH_SIZE
        )
        dataloaders = {'train': train_loader, 'val': val_loader}
        print(f"Training dataset size: {len(train_ds)} samples")
        print(f"Validation dataset size: {len(val_ds)} samples")
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check dataset paths and structure in config.py and dataset.py.")
        return

    # --- 2. Load Model ---
    print("\n--- 2. Loading Model ---")
    try:
        model = models.get_model(model_to_train, config.NUM_CLASSES, pretrained=config.USE_PRETRAINED) # Use config for pretrained
        model = model.to(config.DEVICE)
        print(f"Model '{model_to_train}' loaded successfully on {config.DEVICE}.")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params:,}")
    except ValueError as e: # Specific error from models.get_model if name is invalid
        print(f"Model Loading Error: {e}")
        print(f"Ensure '{model_to_train}' is defined in models.py and supported.")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading the model '{model_to_train}': {e}")
        return

    # --- 3. Define Loss and Optimizer ---
    print("\n--- 3. Defining Loss, Optimizer, and Scheduler ---")
    criterion = nn.CrossEntropyLoss()
    print(f"Loss function: CrossEntropyLoss")

    # Example: Use different learning rates for different parts of the model if desired
    # For now, optimizing all parameters with the same LR
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    print(f"Optimizer: Adam (LR={config.LEARNING_RATE}, Weight Decay={config.WEIGHT_DECAY})")

    # Decay LR by a factor of gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)
    print(f"LR Scheduler: StepLR (Step Size={config.SCHEDULER_STEP_SIZE}, Gamma={config.SCHEDULER_GAMMA})")

    # --- 4. Train ---
    print("\n--- 4. Starting Model Training ---")
    trained_model, history = train_model(
        model,
        model_to_train,
        criterion,
        optimizer,
        exp_lr_scheduler,
        dataloaders,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE
    )

    # --- 5. Plot training history ---
    print("\n--- 5. Plotting Training History ---")
    try:
        import matplotlib.pyplot as plt
        plt.style.use('ggplot') # Using a style for better looking plots
        
        epochs_range = range(1, len(history['train_loss']) + 1)

        fig, axs = plt.subplots(1, 3, figsize=(20, 5)) # Added LR plot

        # Plot Loss
        axs[0].plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', linestyle='-')
        axs[0].plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o', linestyle='-')
        axs[0].set_title(f'{model_to_train} - Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].grid(True)

        # Plot Accuracy
        axs[1].plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o', linestyle='-')
        axs[1].plot(epochs_range, history['val_acc'], label='Validation Accuracy', marker='o', linestyle='-')
        axs[1].set_title(f'{model_to_train} - Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_ylim(0, 1.05) # Set y-axis limit for accuracy

        # Plot Learning Rate
        axs[2].plot(epochs_range, history['lr'], label='Learning Rate', marker='o', linestyle='-', color='green')
        axs[2].set_title(f'{model_to_train} - Learning Rate')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Learning Rate')
        axs[2].legend()
        axs[2].grid(True)

        plt.suptitle(f'Training History for {model_to_train.upper()}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        
        plot_filename = os.path.join(config.MODEL_SAVE_DIR, f"{model_to_train}_training_history.png")
        plt.savefig(plot_filename)
        print(f"Training history plot saved to: {plot_filename}")
        # plt.show() # Uncomment to display plot immediately
        plt.close(fig) # Close the figure to free memory

    except ImportError:
        print("Matplotlib not found. Skipping plotting training history.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

    print(f"\n--- Training and Post-processing finished for {model_to_train.upper()} ---")


if __name__ == '__main__':
    # Ensure MODEL_NAMES in config.py is a list of strings
    if not hasattr(config, 'MODEL_NAMES') or not isinstance(config.MODEL_NAMES, list):
        print("Error: config.MODEL_NAMES is not defined or not a list in config.py.")
        print("Please define it e.g., MODEL_NAMES = ['resnet18', 'resnet50']")
        exit()
        
    # Check for other necessary configs (examples)
    required_configs = ['BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS', 'DEVICE', 
                        'MODEL_SAVE_DIR', 'NUM_CLASSES', 'USE_PRETRAINED', 
                        'WEIGHT_DECAY', 'SCHEDULER_STEP_SIZE', 'SCHEDULER_GAMMA']
    missing_configs = [attr for attr in required_configs if not hasattr(config, attr)]
    if missing_configs:
        print(f"Error: Missing required attributes in config.py: {', '.join(missing_configs)}")
        exit()


    # You can use argparse to select models, or loop through config.MODEL_NAMES
    # For simplicity, this example loops through models defined in config.py
    
    parser = argparse.ArgumentParser(description="Train specified fault detection models.")
    parser.add_argument(
        "--models",
        nargs="+", # Allows multiple model names
        default=config.MODEL_NAMES, # Default to all models in config if none specified
        choices=config.MODEL_NAMES, # Restrict choices to those in config
        help=f"Specify which model(s) to train from the list: {', '.join(config.MODEL_NAMES)}"
    )
    args = parser.parse_args()

    print(f"Selected models for training: {args.models}")

    for model_name_to_train in args.models:
        # The check for model_name_to_train in config.MODEL_NAMES is implicitly handled by argparse choices
        main(model_name_to_train)