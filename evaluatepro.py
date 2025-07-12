# fault_detection_pipeline/evaluate.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import os
import time
import datetime # For timestamped output folder
import numpy as np
import pandas as pd # For saving metrics to CSV
import math # For calculating subplot grid
import glob # For finding model files

# Import plotting libraries safely
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    try:
        import seaborn as sns
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False
        print("Warning: Seaborn not found. Some plots might use basic Matplotlib.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    SEABORN_AVAILABLE = False
    print("Warning: Matplotlib not found. All plot generation will be skipped. Install with: pip install matplotlib seaborn")

# Import scikit-learn metrics
try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix as sk_confusion_matrix,
        accuracy_score,
        roc_curve,
        auc,
        roc_auc_score, # <--- ADD THIS LINE
        precision_recall_curve,
        average_precision_score,
        matthews_corrcoef
    )
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Advanced metrics and plots will be skipped. Install with: pip install scikit-learn")

import config
import models
import dataset

# --- Function to load models (MODIFIED) ---
def load_trained_models(model_names, num_classes, device='cpu'):
    """Loads all specified trained models (best and last versions) from saved weights."""
    loaded_models = {}
    print("\n--- Loading Trained Models ---")
    
    for name in model_names:
        # --- Find and load the BEST model ---
        best_model_pattern = os.path.join(config.MODEL_SAVE_DIR, f"{name}_best_val_acc_*.pth")
        best_model_files = glob.glob(best_model_pattern)
        
        if best_model_files:
            # Sort by accuracy (descending) just in case there are multiple, and pick the highest.
            # We extract accuracy from the filename.
            def get_acc_from_path(filepath):
                try:
                    # Extracts float after '_acc_' and before '.pth'
                    return float(os.path.basename(filepath).split('_acc_')[1].replace('.pth', ''))
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse accuracy from {os.path.basename(filepath)}. Using 0.0.")
                    return 0.0 # Fallback
            
            best_model_files.sort(key=get_acc_from_path, reverse=True)
            model_path_to_load = best_model_files[0]
            model_key = f"{name}_best" # New key for the dictionary
            try:
                print(f"Loading model: {model_key} from {os.path.basename(model_path_to_load)}")
                model = models.get_model(name, num_classes, pretrained=False) # Use base name 'name'
                model.load_state_dict(torch.load(model_path_to_load, map_location=device))
                model.to(device)
                model.eval()
                loaded_models[model_key] = model
                print(f"Successfully loaded {model_key}.")
            except Exception as e:
                print(f"Warning: Error loading model {model_key}: {e}. Skipping.")
        else:
            print(f"Warning: Best model weights file not found for {name} using pattern '{os.path.basename(best_model_pattern)}'. Skipping.")

        # --- Find and load the LAST model ---
        # last_model_pattern = os.path.join(config.MODEL_SAVE_DIR, f"{name}_epoch_*_last.pth")
        # last_model_files = glob.glob(last_model_pattern)

        # if last_model_files:
        #     # If multiple exist, pick the one with the highest epoch number.
        #     def get_epoch_from_path(filepath):
        #         try:
        #             # Extracts int between '_epoch_' and '_last.pth'
        #             return int(os.path.basename(filepath).split('_epoch_')[1].split('_last.pth')[0])
        #         except (IndexError, ValueError):
        #             print(f"Warning: Could not parse epoch from {os.path.basename(filepath)}. Using 0.")
        #             return 0 # Fallback
            
        #     last_model_files.sort(key=get_epoch_from_path, reverse=True)
        #     model_path_to_load = last_model_files[0]
        #     model_key = f"{name}_last" # New key for the dictionary
        #     try:
        #         print(f"Loading model: {model_key} from {os.path.basename(model_path_to_load)}")
        #         model = models.get_model(name, num_classes, pretrained=False) # Use base name 'name'
        #         model.load_state_dict(torch.load(model_path_to_load, map_location=device))
        #         model.to(device)
        #         model.eval()
        #         loaded_models[model_key] = model
        #         print(f"Successfully loaded {model_key}.")
        #     except Exception as e:
        #         print(f"Warning: Error loading model {model_key}: {e}. Skipping.")
        # else:
        #      print(f"Warning: Last model weights file not found for {name} using pattern '{os.path.basename(last_model_pattern)}'. Skipping.")

    if not loaded_models:
        print("\nError: No trained models were successfully loaded.")
    else:
        print(f"\nSuccessfully loaded {len(loaded_models)} model versions: {list(loaded_models.keys())}")
        
    print("--- Model Loading Complete ---")
    return loaded_models

# --- Function for single image prediction (Unchanged) ---
def predict_single_image(image_path, loaded_models, transform, device, class_names):
    """Makes predictions on a single image using multiple loaded models."""
    if not loaded_models: return None
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error opening or transforming image {image_path}: {e}")
        return None
    results = []
    with torch.no_grad():
        for model_name, model in loaded_models.items():
            try:
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_class = class_names[predicted_idx.item()]
                all_probs_dict = {class_names[i]: f"{prob.item():.4f}" for i, prob in enumerate(probabilities)}
                results.append({'model_name': model_name, 'predicted_class': predicted_class, 'confidence': confidence.item(), 'all_probabilities': all_probs_dict})
            except Exception as e:
                print(f"Error during prediction with model {model_name}: {e}")
                results.append({'model_name': model_name, 'predicted_class': 'Error', 'confidence': 0.0, 'all_probabilities': {}})
    return results

# --- Function for evaluating on Test Set (Significantly Enhanced) ---
def evaluate_models_on_test_set(loaded_models, test_dir, transform, batch_size, device, num_classes, class_names, output_dir):
    """
    Evaluates multiple models on a given test dataset directory, calculating
    a comprehensive set of metrics and collecting data for plots.

    Returns:
        dict: A dictionary containing detailed evaluation results for each model.
              {model_name: {
                  'overall_accuracy': float,
                  'confusion_matrix': np.array,
                  'classification_report_dict': dict, # from sklearn
                  'classification_report_str': str,
                  'per_class_specificity': list[float],
                  'mcc': float, # Matthews Correlation Coefficient
                  'roc_auc_per_class': list[float],
                  'roc_auc_macro': float,
                  'roc_auc_weighted': float,
                  'pr_auc_per_class': list[float], # Precision-Recall AUC
                  'pr_auc_macro': float,
                  'pr_auc_weighted': float,
                  'y_true': np.array, # For saving/later use
                  'y_pred': np.array,
                  'y_prob': np.array # (num_samples, num_classes) probabilities
              }}
    """
    if not loaded_models: return None
    if not SKLEARN_AVAILABLE:
        print("Skipping detailed evaluation as scikit-learn is not available.")
        return None

    print(f"\n--- Loading Test Data from: {test_dir} ---")
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory not found at '{test_dir}'.")
        return None
    try:
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        # Ensure class_names from config match those found by ImageFolder
        # This is crucial for correct metric interpretation
        if test_dataset.classes != class_names:
            print("Warning: Class mismatch between test dataset folders and config.py!")
            print(f"  Dataset classes: {test_dataset.classes}")
            print(f"  Config classes: {class_names}")
            # Potentially map or raise an error, for now, we'll proceed but metrics might be mislabeled
        if len(test_dataset) == 0: print(f"Error: No images found in '{test_dir}'."); return None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
        print(f"Test dataset loaded: {len(test_dataset)} images ({num_classes} classes), {len(test_loader)} batches.")
    except Exception as e: print(f"Error loading test dataset: {e}"); return None

    print("\n--- Starting Detailed Evaluation on Test Set ---")
    evaluation_results = {}

    for model_name, model in loaded_models.items():
        print(f"Evaluating model: {model_name}...")
        model.eval()
        start_time = time.time()

        all_labels_list = []
        all_preds_list = []
        all_probs_list = [] # For ROC/PR curves

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_labels_list.append(labels.cpu().numpy())
                all_preds_list.append(preds.cpu().numpy())
                all_probs_list.append(probabilities.cpu().numpy())

        y_true = np.concatenate(all_labels_list)
        y_pred = np.concatenate(all_preds_list)
        y_prob = np.concatenate(all_probs_list) # Shape: (n_samples, n_classes)

        # Save raw outputs
        np.save(os.path.join(output_dir, f"{model_name}_y_true.npy"), y_true)
        np.save(os.path.join(output_dir, f"{model_name}_y_pred.npy"), y_pred)
        np.save(os.path.join(output_dir, f"{model_name}_y_prob.npy"), y_prob)

        # 1. Overall Accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)

        # 2. Confusion Matrix
        cm = sk_confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)) # Ensure labels cover all classes

        # 3. Classification Report (Precision, Recall, F1-score, Support)
        # Use target_names for readable report, ensure it aligns with num_classes
        target_names_for_report = class_names if len(class_names) == num_classes else [f"Class {i}" for i in range(num_classes)]
        report_dict = classification_report(y_true, y_pred, target_names=target_names_for_report, output_dict=True, zero_division=0)
        report_str = classification_report(y_true, y_pred, target_names=target_names_for_report, zero_division=0)
        
        # Save classification report to a text file
        report_path = os.path.join(output_dir, "classification_reports")
        os.makedirs(report_path, exist_ok=True)
        with open(os.path.join(report_path, f"{model_name}_classification_report.txt"), "w") as f:
            f.write(f"Classification Report for Model: {model_name}\n\n")
            f.write(report_str)
        print(f"Classification report for {model_name} saved to {report_path}")


        # 4. Per-Class Specificity
        per_class_specificity = []
        for i in range(num_classes):
            tp_i = cm[i, i]
            fp_i = cm[:, i].sum() - tp_i
            fn_i = cm[i, :].sum() - tp_i
            tn_i = cm.sum() - (tp_i + fp_i + fn_i)

            specificity = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0
            per_class_specificity.append(specificity)

        # 5. Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(y_true, y_pred)

        # 6. ROC AUC Scores
        y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
        roc_auc_per_class = []
        roc_auc_macro = float('nan')
        roc_auc_weighted = float('nan')

        if num_classes > 1 and y_true_binarized.shape[1] == y_prob.shape[1]:
            for i in range(num_classes):
                if y_true_binarized[:, i].sum() > 0 and (1 - y_true_binarized[:, i]).sum() > 0:
                    roc_auc_per_class.append(roc_auc_score(y_true_binarized[:, i], y_prob[:, i]))
                else:
                    roc_auc_per_class.append(float('nan'))
            try:
                roc_auc_macro = roc_auc_score(y_true_binarized, y_prob, average="macro", multi_class="ovr")
                roc_auc_weighted = roc_auc_score(y_true_binarized, y_prob, average="weighted", multi_class="ovr")
            except ValueError as e:
                print(f"Warning: Could not compute macro/weighted ROC AUC for {model_name}: {e}")
        elif num_classes == 1: # Binary (or handle as special case)
            prob_positive_class = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.ravel()
            if y_true.sum() > 0 and (1-y_true).sum() > 0:
                roc_auc = roc_auc_score(y_true, prob_positive_class)
                roc_auc_per_class.append(roc_auc)
                roc_auc_macro = roc_auc_weighted = roc_auc

        # 7. Precision-Recall (PR) AUC Scores
        pr_auc_per_class = []
        pr_auc_macro = float('nan')
        pr_auc_weighted = float('nan')

        if num_classes > 1 and y_true_binarized.shape[1] == y_prob.shape[1]:
            for i in range(num_classes):
                if y_true_binarized[:, i].sum() > 0:
                    precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_prob[:, i])
                    pr_auc_per_class.append(auc(recall, precision))
                else:
                    pr_auc_per_class.append(float('nan'))
            try:
                pr_auc_macro = average_precision_score(y_true_binarized, y_prob, average="macro")
                pr_auc_weighted = average_precision_score(y_true_binarized, y_prob, average="weighted")
            except ValueError as e:
                print(f"Warning: Could not compute macro/weighted PR AUC for {model_name}: {e}")
        elif num_classes == 1: # Binary
            prob_positive_class = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.ravel()
            if y_true.sum() > 0:
                precision, recall, _ = precision_recall_curve(y_true, prob_positive_class)
                pr_auc = auc(recall, precision)
                pr_auc_per_class.append(pr_auc)
                pr_auc_macro = pr_auc_weighted = pr_auc

        elapsed_time = time.time() - start_time

        evaluation_results[model_name] = {
            'overall_accuracy': overall_accuracy,
            'confusion_matrix': cm,
            'classification_report_dict': report_dict,
            'classification_report_str': report_str,
            'per_class_recall': [report_dict[c]['recall'] for c in target_names_for_report if isinstance(report_dict.get(c), dict)],
            'per_class_precision': [report_dict[c]['precision'] for c in target_names_for_report if isinstance(report_dict.get(c), dict)],
            'per_class_f1_score': [report_dict[c]['f1-score'] for c in target_names_for_report if isinstance(report_dict.get(c), dict)],
            'per_class_specificity': per_class_specificity,
            'mcc': mcc,
            'roc_auc_per_class': roc_auc_per_class,
            'roc_auc_macro': roc_auc_macro,
            'roc_auc_weighted': roc_auc_weighted,
            'pr_auc_per_class': pr_auc_per_class,
            'pr_auc_macro': pr_auc_macro,
            'pr_auc_weighted': pr_auc_weighted,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'evaluation_time_seconds': elapsed_time
        }

        print(f"   Model: {model_name}")
        print(f"     Overall Accuracy: {overall_accuracy:.4f}")
        print(f"     Macro F1-Score: {report_dict.get('macro avg', {}).get('f1-score', float('nan')):.4f}")
        print(f"     Weighted F1-Score: {report_dict.get('weighted avg', {}).get('f1-score', float('nan')):.4f}")
        print(f"     Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"     Macro ROC AUC: {roc_auc_macro:.4f}")
        print(f"     Macro PR AUC (Average Precision): {pr_auc_macro:.4f}")
        print(f"     Confusion Matrix (raw counts):\n{cm}")
        print(f"     Evaluation Time: {elapsed_time:.2f} seconds"); print("-" * 40)

    print("--- Detailed Evaluation Complete ---")
    return evaluation_results

# --- Plotting Functions (Existing and New) ---

def plot_metric_comparison(results, class_names, metric_key, metric_name, output_filename):
    """
    Generates a grouped bar chart comparing a specific per-class metric for multiple models.
    e.g., metric_key='per_class_recall', metric_name='Recall'
    """
    if not MATPLOTLIB_AVAILABLE or not results or not SKLEARN_AVAILABLE: return
    model_names = list(results.keys())
    num_models = len(model_names)
    num_classes = len(class_names)

    metric_data = []
    valid_model_names = [] # Keep track of models that have data
    for m in model_names:
        if metric_key in results[m] and results[m][metric_key] is not None and len(results[m][metric_key]) == num_classes:
            metric_data.append(results[m][metric_key])
            valid_model_names.append(m)
        else:
            print(f"Warning: Metric '{metric_key}' not found or mismatched for model {m}. Skipping in plot.")
            
    if not valid_model_names:
        print(f"No valid data to plot for {metric_name}. Skipping plot generation.")
        return
        
    metric_data = np.array(metric_data)
    num_models = len(valid_model_names) # Update num_models

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(max(12, num_classes * 1.5), 7))
    x = np.arange(num_classes)
    width = 0.8 / num_models if num_models > 0 else 0.8
    
    for i, model_name in enumerate(valid_model_names):
        current_offset = width * (i - (num_models - 1) / 2)
        ax.bar(x + current_offset, metric_data[i], width, label=model_name)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel('Fault Class', fontsize=12)
    ax.set_title(f'Model Comparison: Per-Class {metric_name} on Test Set', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(title="Models", bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"{metric_name} summary plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving {metric_name} plot: {e}")
    plt.close(fig)


def plot_confusion_matrices(results, class_names, output_filename):
    if not MATPLOTLIB_AVAILABLE or not results or not SKLEARN_AVAILABLE: return
    model_names = list(results.keys())
    num_models = len(model_names)
    if num_models == 0: return

    ncols = math.ceil(math.sqrt(num_models))
    nrows = math.ceil(num_models / ncols)
    figsize_width = max(10, ncols * 6)
    figsize_height = max(8, nrows * 5.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_width, figsize_height), squeeze=False)
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, y=1.0 if nrows > 1 else 1.05)

    axes_flat = axes.flatten()
    plot_index = 0
    for model_name in model_names:
        if plot_index < len(axes_flat):
            ax = axes_flat[plot_index]
            cm = results[model_name]['confusion_matrix']
            if SEABORN_AVAILABLE:
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=class_names, yticklabels=class_names,
                            annot_kws={"size": 8}, cbar=False)
            else:
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2. else "black", size=8)
                ax.set_xticks(np.arange(len(class_names)))
                ax.set_yticks(np.arange(len(class_names)))
                ax.set_xticklabels(class_names)
                ax.set_yticklabels(class_names)

            ax.set_title(f'Model: {model_name}', fontsize=12)
            ax.set_ylabel('True Class', fontsize=10)
            ax.set_xlabel('Predicted Class', fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
            plot_index += 1
        else: break

    for i in range(plot_index, len(axes_flat)): axes_flat[i].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97 if nrows > 1 else 0.95])
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices plot saved to: {output_filename}")
    except Exception as e: print(f"Error saving confusion matrix plot: {e}")
    plt.close(fig)

def plot_roc_curves(results, class_names, num_classes, output_filename):
    if not MATPLOTLIB_AVAILABLE or not results or not SKLEARN_AVAILABLE: return
    model_names = list(results.keys())
    num_models = len(model_names)
    if num_models == 0: return

    ncols = math.ceil(math.sqrt(num_models))
    nrows = math.ceil(num_models / ncols)
    figsize_width = max(10, ncols * 7)
    figsize_height = max(8, nrows * 6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_width, figsize_height), squeeze=False)
    fig.suptitle('ROC Curves Comparison (Per-Class and Macro Average)', fontsize=16, y=1.0 if nrows > 1 else 1.05)
    axes_flat = axes.flatten()
    plot_idx = 0

    for model_name in model_names:
        if plot_idx < len(axes_flat):
            ax = axes_flat[plot_idx]
            y_true = results[model_name]['y_true']
            y_prob = results[model_name]['y_prob']
            y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))

            # Plot per-class ROC
            for i in range(num_classes):
                if y_true_binarized[:, i].sum() > 0 and (1 - y_true_binarized[:, i]).sum() > 0:
                    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=1.5, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
                else:
                    ax.plot([0,1],[0,1], linestyle='--', lw=1.5, label=f'Class {class_names[i]} (N/A)')

            # Plot Macro-average ROC
            if num_classes > 1 and y_true_binarized.shape[1] == y_prob.shape[1]:
                all_fpr = np.unique(np.concatenate([roc_curve(y_true_binarized[:, i], y_prob[:, i])[0] for i in range(num_classes) if (y_true_binarized[:,i].sum()>0 and (1-y_true_binarized[:,i]).sum()>0)]))
                mean_tpr = np.zeros_like(all_fpr)
                valid_classes = 0
                for i in range(num_classes):
                    if y_true_binarized[:, i].sum() > 0 and (1 - y_true_binarized[:, i]).sum() > 0:
                        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
                        mean_tpr += np.interp(all_fpr, fpr, tpr)
                        valid_classes +=1
                if valid_classes > 0:
                    mean_tpr /= valid_classes
                    macro_roc_auc = auc(all_fpr, mean_tpr)
                    ax.plot(all_fpr, mean_tpr, color='navy', linestyle=':', linewidth=3,
                            label=f'Macro Avg (AUC = {macro_roc_auc:.2f})')

            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(f'Model: {model_name}', fontsize=12)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True)
            plot_idx += 1
        else: break

    for i in range(plot_idx, len(axes_flat)): axes_flat[i].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97 if nrows > 1 else 0.95])
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"ROC curves plot saved to: {output_filename}")
    except Exception as e: print(f"Error saving ROC curves plot: {e}")
    plt.close(fig)


def plot_precision_recall_curves(results, class_names, num_classes, output_filename):
    if not MATPLOTLIB_AVAILABLE or not results or not SKLEARN_AVAILABLE: return
    model_names = list(results.keys())
    num_models = len(model_names)
    if num_models == 0: return

    ncols = math.ceil(math.sqrt(num_models))
    nrows = math.ceil(num_models / ncols)
    figsize_width = max(10, ncols * 7)
    figsize_height = max(8, nrows * 6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_width, figsize_height), squeeze=False)
    fig.suptitle('Precision-Recall Curves Comparison (Per-Class)', fontsize=16, y=1.0 if nrows > 1 else 1.05)
    axes_flat = axes.flatten()
    plot_idx = 0

    for model_name in model_names:
        if plot_idx < len(axes_flat):
            ax = axes_flat[plot_idx]
            y_true = results[model_name]['y_true']
            y_prob = results[model_name]['y_prob']
            y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))

            # Plot per-class PR
            for i in range(num_classes):
                if y_true_binarized[:, i].sum() > 0:
                    precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_prob[:, i])
                    ap = average_precision_score(y_true_binarized[:, i], y_prob[:, i])
                    ax.plot(recall, precision, lw=1.5, label=f'Class {class_names[i]} (AP = {ap:.2f})')
                else:
                    ax.plot([0,1],[1,0], linestyle='--', lw=1.5, label=f'Class {class_names[i]} (N/A)')

            macro_ap = results[model_name].get('pr_auc_macro', float('nan'))
            ax.plot([0, 1], [0.5, 0.5], # Example baseline
                    linestyle=':', color='navy', lw=2, label=f'Macro AP = {macro_ap:.2f}')

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_title(f'Model: {model_name}', fontsize=12)
            ax.legend(loc="lower left", fontsize=8)
            ax.grid(True)
            plot_idx += 1
        else: break

    for i in range(plot_idx, len(axes_flat)): axes_flat[i].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97 if nrows > 1 else 0.95])
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curves plot saved to: {output_filename}")
    except Exception as e: print(f"Error saving Precision-Recall curves plot: {e}")
    plt.close(fig)

# --- Function to Save Summary Metrics to CSV ---
def save_metrics_summary_to_csv(results, class_names, num_classes, output_filepath):
    if not results or not SKLEARN_AVAILABLE: return
    
    summary_data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row['Overall Accuracy'] = metrics.get('overall_accuracy')
        row['MCC'] = metrics.get('mcc')

        report_dict = metrics.get('classification_report_dict', {})
        row['Macro Avg Precision'] = report_dict.get('macro avg', {}).get('precision')
        row['Macro Avg Recall'] = report_dict.get('macro avg', {}).get('recall')
        row['Macro Avg F1-Score'] = report_dict.get('macro avg', {}).get('f1-score')
        row['Weighted Avg Precision'] = report_dict.get('weighted avg', {}).get('precision')
        row['Weighted Avg Recall'] = report_dict.get('weighted avg', {}).get('recall')
        row['Weighted Avg F1-Score'] = report_dict.get('weighted avg', {}).get('f1-score')

        row['Macro ROC AUC'] = metrics.get('roc_auc_macro')
        row['Weighted ROC AUC'] = metrics.get('roc_auc_weighted')
        row['Macro PR AUC'] = metrics.get('pr_auc_macro')
        row['Weighted PR AUC'] = metrics.get('pr_auc_weighted')
        
        target_names_for_report = class_names if len(class_names) == num_classes else [f"Class {i}" for i in range(num_classes)]

        for i, c_name in enumerate(target_names_for_report):
            row[f'{c_name} Precision'] = metrics.get('per_class_precision', [])[i] if i < len(metrics.get('per_class_precision', [])) else None
            row[f'{c_name} Recall'] = metrics.get('per_class_recall', [])[i] if i < len(metrics.get('per_class_recall', [])) else None
            row[f'{c_name} F1-Score'] = metrics.get('per_class_f1_score', [])[i] if i < len(metrics.get('per_class_f1_score', [])) else None
            row[f'{c_name} Specificity'] = metrics.get('per_class_specificity', [])[i] if i < len(metrics.get('per_class_specificity', [])) else None
            row[f'{c_name} ROC AUC'] = metrics.get('roc_auc_per_class', [])[i] if i < len(metrics.get('roc_auc_per_class', [])) else None
            row[f'{c_name} PR AUC'] = metrics.get('pr_auc_per_class', [])[i] if i < len(metrics.get('pr_auc_per_class', [])) else None
        
        row['Evaluation Time (s)'] = metrics.get('evaluation_time_seconds')
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    try:
        df.to_csv(output_filepath, index=False, float_format='%.4f')
        print(f"\nDetailed metrics summary saved to: {output_filepath}")
    except Exception as e:
        print(f"Error saving metrics summary CSV: {e}")


# --- Main Execution Block ---
if __name__ == '__main__':
    if not SKLEARN_AVAILABLE:
        print("Critical Error: scikit-learn is required for evaluation. Please install it.")
        exit()

    # --- 1. Create Timestamped Output Directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(config.BASE_DIR, "evaluation_outputs")
    current_run_output_dir = os.path.join(base_output_dir, f"{timestamp}_evaluation")
    os.makedirs(current_run_output_dir, exist_ok=True)
    print(f"All evaluation outputs will be saved in: {current_run_output_dir}")

    # --- 2. Load models ---
    # We pass the base model names (e.g., 'resnet18') and the function
    # will find the '_best' and '_last' versions.
    loaded_models_dict = load_trained_models(config.MODEL_NAMES, config.NUM_CLASSES, config.DEVICE)

    if loaded_models_dict:
        # --- 3. Evaluate models ---
        evaluation_results = evaluate_models_on_test_set(
            loaded_models=loaded_models_dict,
            test_dir=config.TEST_DIR,
            transform=dataset.val_test_transforms, # Make sure this transform is appropriate
            batch_size=config.BATCH_SIZE,
            device=config.DEVICE,
            num_classes=config.NUM_CLASSES,
            class_names=config.CLASSES,
            output_dir=current_run_output_dir # Pass output directory
        )

        # --- 4. Print Summary and Generate Plots/CSVs if results exist ---
        if evaluation_results:
            print("\n--- Overall Performance Summary ---")
            # Sort by a primary metric, e.g., Weighted F1 or Overall Accuracy
            sorted_overall = sorted(
                evaluation_results.items(),
                key=lambda item: item[1].get('classification_report_dict', {}).get('weighted avg', {}).get('f1-score', 0),
                reverse=True
            )
            for model_name, metrics in sorted_overall:
                acc = metrics.get('overall_accuracy', float('nan'))
                macro_f1 = metrics.get('classification_report_dict', {}).get('macro avg', {}).get('f1-score', float('nan'))
                weighted_f1 = metrics.get('classification_report_dict', {}).get('weighted avg', {}).get('f1-score', float('nan'))
                mcc_val = metrics.get('mcc', float('nan'))
                print(f"  Model: {model_name}")
                print(f"    Overall Accuracy: {acc:.4f}")
                print(f"    Macro F1-Score: {macro_f1:.4f}")
                print(f"    Weighted F1-Score: {weighted_f1:.4f}")
                print(f"    MCC: {mcc_val:.4f}")
            print("---------------------------------------------")

            # --- Save Detailed Metrics to CSV ---
            csv_save_path = os.path.join(current_run_output_dir, "detailed_metrics_summary.csv")
            save_metrics_summary_to_csv(evaluation_results, config.CLASSES, config.NUM_CLASSES, csv_save_path)

            if MATPLOTLIB_AVAILABLE:
                # --- Generate Per-Class Recall (Accuracy) Bar Plot ---
                plot_recall_save_path = os.path.join(current_run_output_dir, "evaluation_recall_comparison_plot.png")
                plot_metric_comparison(
                    results=evaluation_results, class_names=config.CLASSES,
                    metric_key='per_class_recall', metric_name='Recall',
                    output_filename=plot_recall_save_path
                )

                # --- Generate Per-Class Precision Bar Plot ---
                plot_precision_save_path = os.path.join(current_run_output_dir, "evaluation_precision_comparison_plot.png")
                plot_metric_comparison(
                    results=evaluation_results, class_names=config.CLASSES,
                    metric_key='per_class_precision', metric_name='Precision',
                    output_filename=plot_precision_save_path
                )

                # --- Generate Per-Class F1-Score Bar Plot ---
                plot_f1_save_path = os.path.join(current_run_output_dir, "evaluation_f1_score_comparison_plot.png")
                plot_metric_comparison(
                    results=evaluation_results, class_names=config.CLASSES,
                    metric_key='per_class_f1_score', metric_name='F1-Score',
                    output_filename=plot_f1_save_path
                )
                
                # --- Generate Confusion Matrices Plot ---
                plot_cm_save_path = os.path.join(current_run_output_dir, "evaluation_confusion_matrices_plot.png")
                plot_confusion_matrices(
                    results=evaluation_results, class_names=config.CLASSES,
                    output_filename=plot_cm_save_path
                )

                # --- Generate ROC Curves Plot ---
                plot_roc_save_path = os.path.join(current_run_output_dir, "evaluation_roc_curves_plot.png")
                plot_roc_curves(
                    results=evaluation_results, class_names=config.CLASSES,
                    num_classes=config.NUM_CLASSES, output_filename=plot_roc_save_path
                )

                # --- Generate Precision-Recall Curves Plot ---
                plot_pr_save_path = os.path.join(current_run_output_dir, "evaluation_pr_curves_plot.png")
                plot_precision_recall_curves(
                    results=evaluation_results, class_names=config.CLASSES,
                    num_classes=config.NUM_CLASSES, output_filename=plot_pr_save_path
                )
            else:
                print("\nPlot generation skipped as Matplotlib is not available.")
        else:
            print("\nEvaluation could not be completed or produced no results.")
    else:
        print("\nCannot run evaluation as no models were loaded.")

    print(f"\nEvaluation script finished. All outputs are in: {current_run_output_dir}")