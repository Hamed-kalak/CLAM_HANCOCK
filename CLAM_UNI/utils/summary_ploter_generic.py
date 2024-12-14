import re
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def setup_experiment_paths(base_directory, exp_name):
    """
    Automatically set up experiment paths, find relevant files, and determine total classes from the base directory.
    Dynamically constructs file paths based on the exp_name pattern.
    
    Parameters:
    - base_directory: The base directory where experiments are located.
    - exp_name: The experiment name, which varies but contains identifiable patterns.
    
    Returns:
    - A tuple containing the text file path, log file path, and total classes.
    """
    
    # Extract total classes from base directory name
    total_classes_match = re.search(r'(\d+)class', base_directory)
    if total_classes_match:
        total_classes = int(total_classes_match.group(1))
    else:
        print("Could not determine total classes from base directory name.")
        return None, None, None

    # Extract patterns from the exp_name to build paths dynamically
    def extract_details_from_exp_name(exp_name):
        # Updated pattern to match "exp_8k_sampled_1_s1" format and capture numeric parts
        pattern = re.compile(r"exp_(\d+k)_sampled_(\d+)_s(\d+)")
        match = pattern.search(exp_name)
        if match:
            return match.groups()
        return None

    details = extract_details_from_exp_name(exp_name)
    if details is None:
        print("Invalid experiment name format.")
        return None, None, None

    sampling, exp_number, series_number = details
    # Adjust the path construction as needed based on the experiment name structure
    directory_name = f'exp_{sampling}_sampled_{exp_number}_s{series_number}'
    text_file_path = os.path.join(base_directory, directory_name, f'experiment_exp_{sampling}_sampled_{exp_number}.txt')

    log_file = None
    for root, dirs, files in os.walk(os.path.join(base_directory, directory_name)):
        for file in files:
            if file.endswith('.de'):  # Assuming log files have '.log' extension
                log_file = os.path.join(root, file)
                break
        if log_file:  # Break the outer loop if log_file is found
            break

    return text_file_path, log_file, total_classes





def process_experiment_data(log_file):
    """
    Dynamically processes experiment data, organizing metrics based on encountered keys.
    
    Parameters:
    - text_file_path: Path to the text file.
    - log_file: Path to the log file.
    
    Returns:
    - A dictionary with all metrics organized by their keys.
    """
    # Initialize a dictionary to store metrics
    metrics = {}

    # Function to dynamically update metrics based on log file entries
    def update_metrics(e, v):
        tag = v.tag
        value = v.simple_value
        step = e.step

        # Create a nested structure if not already present
        if tag not in metrics:
            metrics[tag] = []

        # Append the value and step to the appropriate key
        metrics[tag].append((step, value))

    # Check if the path is None
    if log_file is None:
        raise ValueError("log_file path is None")

    # Process the log file
    for e in tf.compat.v1.train.summary_iterator(log_file):
        for v in e.summary.value:
            update_metrics(e, v)

    return metrics


def plot_text_data(text_file_path, ax, exp_name):
    """
    Plots key-value pairs from a dictionary as text in a given AxesSubplot.

    Parameters:
    - data_dict: Dictionary containing the data to be displayed as text.
    - ax: The matplotlib AxesSubplot to plot the text on.
    - exp_name: Experiment name to be used as the plot title.
    """

    with open(text_file_path, 'r') as file:
        text_data = file.read()

    # Parse text data into a dictionary
    data_dict = eval(text_data)
    ax.set_facecolor('white')
    for i, (key, value) in enumerate(data_dict.items()):
        ax.text(0.5, 1 - i*0.05, f"{key}: {value}", transform=ax.transAxes, ha='center', va='top')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(exp_name, fontsize=20)
    ax.axis('off')


def plot_losses(metrics_data, ax=None):
    """
    Plots training, validation, validation instance, and clustering losses on a single AxesSubplot.

    Parameters:
    - metrics_data: A dictionary containing the loss data.
    - ax: The matplotlib AxesSubplot to plot the losses on. If None, uses the current axis.
    """
    if ax is None:
        ax = plt.gca()

    # Plot Training Loss
    if 'train/loss' in metrics_data:
        steps, values = zip(*metrics_data['train/loss'])
        ax.plot(steps, values, label='Training Loss', marker='o', linestyle='-', markersize=4)

    # Plot Validation Loss
    if 'val/loss' in metrics_data:
        steps, values = zip(*metrics_data['val/loss'])
        ax.plot(steps, values, label='Validation Loss', marker='o', linestyle='-', markersize=4)

    # Plot Validation Instance Loss
    if 'val/inst_loss' in metrics_data:
        steps, values = zip(*metrics_data['val/inst_loss'])
        ax.plot(steps, values, label='Validation Instance Loss', marker='o', linestyle='-', markersize=4)

    # Plot Clustering Loss
    if 'train/clustering_loss' in metrics_data:
        steps, values = zip(*metrics_data['train/clustering_loss'])
        ax.plot(steps, values, label='Clustering Loss', marker='x', linestyle='-', markersize=4)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Losses Over Time')
    ax.grid(True)
    ax.legend()


def plot_training_accuracies_and_overall(metrics_data, total_classes, ax=None):
    """
    Plots individual class training accuracies and overall training accuracy on a single AxesSubplot.

    Parameters:
    - metrics_data: A dictionary containing the accuracy and error data.
    - total_classes: The total number of classes.
    - ax: The matplotlib AxesSubplot to plot the accuracies on. If None, uses the current axis.
    """
    if ax is None:
        ax = plt.gca()

    # Plot individual class accuracies
    for i in range(total_classes):
        key = f'train/class_{i}_acc'
        if key in metrics_data:
            steps, values = zip(*metrics_data[key])
            ax.plot(steps, values, label=f'Class {i} Accuracy', marker='o', linestyle='-', markersize=3)

    # Plot overall training accuracy derived from training error, if available
    if 'train/error' in metrics_data:
        error_steps, error_values = zip(*metrics_data['train/error'])
        overall_accuracy = [1 - error for error in error_values]
        ax.plot(error_steps, overall_accuracy, label='Overall Training Accuracy', color='black', linestyle='--',linewidth=1,marker='.')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracies Over Time')
    ax.set_ylim(0, 1.0) 
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.grid(True)
    ax.legend()



def plot_validation_accuracies_and_overall(metrics_data, total_classes, ax=None):
    """
    Plots individual class validation accuracies and overall validation accuracy on a single AxesSubplot.

    Parameters:
    - metrics_data: A dictionary containing the accuracy and error data.
    - total_classes: The total number of classes.
    - ax: The matplotlib AxesSubplot to plot the accuracies on. If None, uses the current axis.
    """
    if ax is None:
        ax = plt.gca()
        

    # Plot individual class accuracies
    for i in range(total_classes):
        key = f'val/class_{i}_acc'
        if key in metrics_data:
            steps, values = zip(*metrics_data[key])
            ax.plot(steps, values, label=f'Class {i} Validation Accuracy', marker='o', linestyle='-', markersize=3)

    # Plot overall validation accuracy derived from validation error, if available
    if 'val/error' in metrics_data:
        error_steps, error_values = zip(*metrics_data['val/error'])
        overall_accuracy = [1 - error for error in error_values]
        ax.plot(error_steps, overall_accuracy, label='Overall Validation Accuracy', color='black', linestyle='--',linewidth=1,marker='.')


    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracies Over Time')
    ax.set_ylim(0, 1.0) 
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.grid(True)
    ax.legend()


def plot_final_metrics(metrics_data, ax=None):
    """
    Plots individual class validation accuracies, overall validation accuracy, validation errors, and AUC scores as a bar plot.

    Parameters:
    - metrics_data: A dictionary containing the accuracy, error, and AUC data.
    - ax: The matplotlib AxesSubplot to plot the metrics on. If None, uses the current axis.
    """
    if ax is None:
        ax = plt.gca()

    # Define the metrics to plot
    metrics_to_plot = {
        'Test Accuracy': 'final/test_acc',
        'Validation Accuracy': 'final/val_acc',
        'Validation Error': 'final/val_error',
        'Test Error': 'final/test_error',
        'Validation AUC': 'final/val_auc',
        'Test AUC': 'final/test_auc'
    }

    # Extract metric names and values
    metric_values = []
    for key in metrics_to_plot.values():
        data = metrics_data.get(key)
        if isinstance(data, list) and data and isinstance(data[0], (list, tuple)) and len(data[0]) > 1:
            metric_value = data[0][-1]  # Assuming you want the last value of the first entry
        else:
            metric_value = np.nan
        metric_values.append(metric_value)

    # Create bar plot
    ax.bar(list(metrics_to_plot.keys()), metric_values, color=['blue', 'orange', 'green', 'red', 'purple', 'pink'])

    # Set labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Metric Value')
    ax.set_title('Final Metrics')

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)

    # Show plot
    plt.tight_layout()


import seaborn as sns

def plot_cm(cm, epoch, tag, ax=None):
    if cm is None:
        print(f"Skipping plot for {tag} - No data available")
        return
    
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='viridis', cbar=True , square=True, linecolor='white')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(tag + f' on Epoch {epoch}')

from sklearn.metrics import auc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_overall_roc(roc_data, ax=None):
    if ax is None:
        ax = plt.gca()
    if roc_data == None:
        return
    
    # Handle the binary class case
    if len(roc_data.keys()) == 2:
        # Assume two classes are '0' and '1', adjust based on your actual class labels
        class_fpr, class_tpr, _ = roc_data['1']['fpr'], roc_data['1']['tpr'], roc_data['1']['thresholds']
        roc_auc = auc(class_fpr, class_tpr)
        ax.plot(class_fpr, class_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        # Handle the multi-class case
        # Create a unified set of all FPR values across all classes for interpolation
        all_fpr = np.unique(np.concatenate([roc_data[i]['fpr'] for i in roc_data.keys()]))
        
        # Interpolate TPR values for each class over the unified FPR array
        mean_tpr = np.zeros_like(all_fpr)
        for i in roc_data.keys():
            mean_tpr += np.interp(all_fpr, roc_data[i]['fpr'], roc_data[i]['tpr'])
        
        # Compute the average TPR at each FPR point
        mean_tpr /= len(roc_data.keys())
        
        # Compute macro-average AUC
        roc_auc_macro = auc(all_fpr, mean_tpr)
        
        # Plot the macro-average ROC curve
        ax.plot(all_fpr, mean_tpr, label=f'Macro-average ROC curve (area = {roc_auc_macro:.2f})', color='navy', linestyle=':', linewidth=4)
        
        # Optional: Plot each class's ROC curve for reference
        for i in roc_data.keys():
            class_fpr, class_tpr, _ = roc_data[i]['fpr'], roc_data[i]['tpr'], roc_data[i]['thresholds']
            class_auc = auc(class_fpr, class_tpr)
            ax.plot(class_fpr, class_tpr, lw=1, alpha=0.5, label=f'ROC curve class {i} (area = {class_auc:.2f})')
    
    # Plotting details
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Overall Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.show()




    



def plot_all_subplots(metrics_data, total_classes, exp_name,text_file_path, cm_data, epoch,roc,ax=None):
    # If ax is not provided, create a new figure with subplots
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(25, 10))  # Adjusted for clarity and layout
    else:
        fig = ax[0, 0].get_figure()

    plot_text_data(text_file_path, ax=ax[0,0], exp_name=exp_name)
    plot_losses(metrics_data, ax=ax[0,1])
    plot_training_accuracies_and_overall(metrics_data, total_classes, ax=ax[0,2])
    plot_validation_accuracies_and_overall(metrics_data, total_classes, ax=ax[0,3])

    cm_v , cm_t = cm_data

    plot_final_metrics(metrics_data, ax=ax[1,0])  # Adjust according to your subplot arrangement

    plot_cm(cm_v, epoch,  tag="Validation Confusion Matrix", ax=ax[1, 1])
    plot_cm(cm_t,epoch, tag="Test Confusion Matrix", ax=ax[1, 2])

    plot_overall_roc(roc, ax=ax[1, 3])
    


    plt.tight_layout()
    plt.show()
    