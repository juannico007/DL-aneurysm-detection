"""
Metrics Visualization Script for Aneurysm Detection Models

This script loads multiple trained models and creates a 2x2 grid plot showing:
- neg_prob_max (negative probability maximum)
- peak_err (peak error)
- dice (Dice coefficient)
- legend (showing which color/line corresponds to which model)

Configuration:
- Edit MODEL_CONFIGS to specify models to compare
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# =====================================================================
# CONFIGURATION
# =====================================================================

# Models to compare - add/remove model directories as needed
MODEL_CONFIGS = [
    "cloud_models/final_models/best_model",
    "cloud_models/final_models/new-loss-2_curriculum-learning-alpha-5",
    "cloud_models/final_models/final_002",
    "cloud_models/final_models/final_005",
    "cloud_models/final_models/final_006",
    "cloud_models/final_models/final_009",
    "cloud_models/final_models/final_004",
]

OUTPUT_CONFIG = {
    "output_file": "metrics_comparison_grid.png",
    "figsize": (20, 12),  # Wider to accommodate 3 columns
    "dpi": 150,
}

# =====================================================================
# MODEL LOADING UTILITIES
# =====================================================================

def load_model_history(model_dir: Path) -> tuple[str, Dict, Dict]:
    """
    Load history from a model directory.
    
    Returns:
        (model_name, hyperparameters, history)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load hyperparameters
    hp_path = model_dir / "hyperparameters.json"
    hyperparams = {}
    if hp_path.exists():
        with hp_path.open() as f:
            hyperparams = json.load(f)
    
    # Load history
    history_path = model_dir / "history.pickle"
    history = {}
    if history_path.exists():
        with history_path.open("rb") as f:
            history = pickle.load(f)
    else:
        print(f"  [Warning] No history.pickle found for {model_dir.name}")
    
    return model_dir.name, hyperparams, history


# =====================================================================
# PLOTTING FUNCTIONS
# =====================================================================

def plot_metric_subplot(ax, histories: Dict[str, Dict], metric_name: str, 
                        title: str, ylabel: str, colors, name_map: Dict[str, str],
                        exclude_models: List[str] = None):
    """
    Plot a single metric on a subplot.
    
    Args:
        ax: Matplotlib axis
        histories: Dict of {model_name: history_dict}
        metric_name: Name of the metric (without train_/val_ prefix)
        title: Plot title
        ylabel: Y-axis label
        colors: Color map for different models
        name_map: Mapping from model directory name to display name
        exclude_models: List of display names to exclude from this plot
    """
    if exclude_models is None:
        exclude_models = []
    
    for idx, (model_name, history) in enumerate(histories.items()):
        display_name = name_map.get(model_name, model_name)
        
        # Skip excluded models
        if display_name in exclude_models:
            continue
        
        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"
        
        # Plot training metric
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 
                   label=f"{display_name} (train)", 
                   color=colors[idx], 
                   linestyle='-', 
                   linewidth=2,
                   alpha=0.7)
        
        # Plot validation metric
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 
                   label=f"{display_name} (val)", 
                   color=colors[idx], 
                   linestyle='--', 
                   linewidth=2,
                   alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)


def create_metrics_grid(histories: Dict[str, Dict], output_path: Path, 
                       name_map: Dict[str, str]):
    """
    Create a 2x3 grid plot with neg_prob_max, peak_err, dice, accuracy, ppv, and legend.
    
    Args:
        histories: Dict of {model_name: history_dict}
        output_path: Path to save the output figure
        name_map: Mapping from model directory name to display name
    """
    # Create color palette
    n_models = len(histories)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=OUTPUT_CONFIG["figsize"])
    
    # Plot 1: neg_prob_max (top-left)
    plot_metric_subplot(
        axes[0, 0], histories, "neg_prob_max_series",
        "Negative Probability Max", "Probability", colors, name_map
    )
    
    # Plot 2: peak_err (top-middle)
    plot_metric_subplot(
        axes[0, 1], histories, "peak_err_series",
        "Peak Error", "Error (voxels)", colors, name_map
    )
    
    # Plot 3: dice (top-right)
    plot_metric_subplot(
        axes[0, 2], histories, "dice",
        "Dice Coefficient", "Dice Score", colors, name_map
    )
    
    # Plot 4: accuracy (bottom-left) - excluding specific models
    plot_metric_subplot(
        axes[1, 0], histories, "acc",
        "Accuracy", "Accuracy", colors, name_map,
        exclude_models=["Just Curriculum Semimetric Dice loss", "Without curriculum learning and background loss"]
    )
    
    # Plot 5: ppv (bottom-middle)
    plot_metric_subplot(
        axes[1, 1], histories, "ppv",
        "Positive Predictive Value (PPV)", "PPV", colors, name_map
    )
    
    # Plot 6: Legend (bottom-right)
    axes[1, 2].axis('off')
    
    # Create legend entries
    legend_lines = []
    legend_labels = []
    
    for idx, (model_name, history) in enumerate(histories.items()):
        display_name = name_map.get(model_name, model_name)
        
        # Add solid line for training
        line_train = plt.Line2D([0], [0], color=colors[idx], linewidth=2, 
                               linestyle='-', alpha=0.7)
        legend_lines.append(line_train)
        legend_labels.append(f"{display_name} (train)")
        
        # Add dashed line for validation
        line_val = plt.Line2D([0], [0], color=colors[idx], linewidth=2, 
                             linestyle='--', alpha=0.9)
        legend_lines.append(line_val)
        legend_labels.append(f"{display_name} (val)")
    
    # Place legend in the bottom-right subplot with larger font
    axes[1, 2].legend(legend_lines, legend_labels, 
                     loc='center', 
                     fontsize=14,  # Increased from 10
                     frameon=True,
                     ncol=1)
    axes[1, 2].set_title('Legend', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=OUTPUT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved metrics grid to {output_path}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    print("=" * 70)
    print("Metrics Grid Visualization for Aneurysm Detection Models")
    print("=" * 70)
    
    # Load all model histories
    print("\n[Step 1] Loading model histories...")
    histories = {}
    hyperparams_map = {}
    
    for model_path in MODEL_CONFIGS:
        try:
            model_name, hyperparams, history = load_model_history(Path(model_path))
            if history:
                histories[model_name] = history
                hyperparams_map[model_name] = hyperparams
                print(f"  ✓ Loaded {model_name}")
            else:
                print(f"  ✗ No history found for {model_name}")
        except Exception as e:
            print(f"  ✗ Failed to load {model_path}: {e}")
    
    if not histories:
        print("[Error] No model histories loaded successfully!")
        return
    
    print(f"\n[Step 2] Successfully loaded {len(histories)} model histories")
    
    # Create name mapping (use description from hyperparameters if available)
    name_map = {}
    for model_name, hyperparams in hyperparams_map.items():
        display_name = hyperparams.get("description", model_name)
        name_map[model_name] = display_name
    print(name_map)
    # Generate the metrics grid
    print("\n[Step 3] Generating metrics grid plot...")
    output_path = Path(OUTPUT_CONFIG["output_file"])
    create_metrics_grid(histories, output_path, name_map)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Models compared: {len(histories)}")
    print(f"Output file: {output_path.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
