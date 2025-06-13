# utils/visualizations.py

import matplotlib.pyplot as plt
import numpy as np

def plot_ablation_comparison_chart():
    """
    Generates and returns the bar chart comparing all experimental results.
    """
    # Data from your final analysis document
    experiment_results = {
        'Baseline': 91.77,
        'Fine-Tuned': 93.01,
        'No SE': 92.00,
        'No Conv2': 92.00,
        'AvgPool': 92.00
    }

    model_names = list(experiment_results.keys())
    accuracies = list(experiment_results.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['deepskyblue', 'limegreen', 'lightcoral', 'orange', 'purple']
    bars = ax.bar(model_names, accuracies, color=colors)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Comparison Across All Models', fontsize=16)
    ax.set_ylim(min(accuracies) - 2, max(accuracies) + 2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom', ha='center')

    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig