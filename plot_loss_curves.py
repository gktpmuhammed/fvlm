#!/usr/bin/env python3
"""
Script to parse FVLM training logs and plot loss curves for all organs across epochs.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import seaborn as sns
from pathlib import Path

def parse_training_log(log_file_path):
    """Parse the training log file and extract loss information."""
    
    # Initialize data structures
    epoch_data = defaultdict(lambda: defaultdict(list))
    overall_loss_data = defaultdict(list)
    
    # Define the organs we expect to find
    organs = ['face', 'brain', 'esophagus', 'trachea', 'lung', 'heart', 'kidney', 
              'stomach', 'liver', 'gallbladder', 'pancreas', 'spleen', 'colon', 
              'aorta', 'rib', 'humerus', 'scapula', 'clavicula', 'femur', 'hip', 
              'sacrum', 'gluteus', 'iliopsoas', 'autochthon']
    
    print(f"Parsing log file: {log_file_path}")
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    training_lines = []
    for line in lines:
        # Look for training progress lines
        if 'Train: data epoch:' in line and 'loss:' in line:
            training_lines.append(line.strip())
    
    print(f"Found {len(training_lines)} training progress lines")
    
    # Parse each training line
    for line in training_lines:
        # Extract epoch number
        epoch_match = re.search(r'data epoch: \[(\d+)\]', line)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        
        # Extract overall loss
        loss_match = re.search(r'loss: ([\d.]+)', line)
        if not loss_match:
            continue
        overall_loss = float(loss_match.group(1))
        overall_loss_data[epoch].append(overall_loss)
        
        # Extract organ-specific losses
        for organ in organs:
            pattern = f'{organ}_itc: ([\d.]+)'
            organ_match = re.search(pattern, line)
            if organ_match:
                organ_loss = float(organ_match.group(1))
                epoch_data[epoch][organ].append(organ_loss)
    
    return epoch_data, overall_loss_data, organs

def calculate_epoch_averages(epoch_data, overall_loss_data, organs):
    """Calculate average losses per epoch."""
    
    epochs = sorted(epoch_data.keys())
    
    # Calculate averages for overall loss
    overall_avg = {}
    for epoch in epochs:
        if overall_loss_data[epoch]:
            overall_avg[epoch] = np.mean(overall_loss_data[epoch])
        else:
            overall_avg[epoch] = 0.0
    
    # Calculate averages for organ losses
    organ_avg = {organ: {} for organ in organs}
    for epoch in epochs:
        for organ in organs:
            if epoch_data[epoch][organ]:
                organ_avg[organ][epoch] = np.mean(epoch_data[epoch][organ])
            else:
                organ_avg[organ][epoch] = 0.0
    
    return overall_avg, organ_avg, epochs

def plot_loss_curves(overall_avg, organ_avg, epochs, organs):
    """Create comprehensive loss curve plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Overall loss curve
    plt.subplot(2, 2, 1)
    overall_losses = [overall_avg[epoch] for epoch in epochs]
    plt.plot(epochs, overall_losses, 'b-', linewidth=2, marker='o', markersize=3)
    plt.title('Overall Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: All organ losses on one plot (with active organs only)
    plt.subplot(2, 2, 2)
    active_organs = []
    for organ in organs:
        organ_losses = [organ_avg[organ][epoch] for epoch in epochs]
        # Only plot organs that have non-zero losses at some point
        if max(organ_losses) > 0:
            active_organs.append(organ)
            plt.plot(epochs, organ_losses, linewidth=1.5, marker='o', markersize=2, label=organ, alpha=0.8)
    
    plt.title('All Organ Loss Curves (Active Organs Only)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Top active organs (those with highest max loss)
    plt.subplot(2, 2, 3)
    organ_max_losses = []
    for organ in active_organs:
        organ_losses = [organ_avg[organ][epoch] for epoch in epochs]
        organ_max_losses.append((organ, max(organ_losses)))
    
    # Sort by max loss and take top 8
    organ_max_losses.sort(key=lambda x: x[1], reverse=True)
    top_organs = [organ for organ, _ in organ_max_losses[:8]]
    
    for organ in top_organs:
        organ_losses = [organ_avg[organ][epoch] for epoch in epochs]
        plt.plot(epochs, organ_losses, linewidth=2, marker='o', markersize=3, label=organ)
    
    plt.title('Top 8 Organs by Maximum Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss trends (first vs last 10 epochs comparison)
    plt.subplot(2, 2, 4)
    first_10_epochs = epochs[:10]
    last_10_epochs = epochs[-10:]
    
    organ_trend_data = []
    for organ in active_organs:
        first_10_avg = np.mean([organ_avg[organ][epoch] for epoch in first_10_epochs])
        last_10_avg = np.mean([organ_avg[organ][epoch] for epoch in last_10_epochs])
        improvement = first_10_avg - last_10_avg
        organ_trend_data.append((organ, first_10_avg, last_10_avg, improvement))
    
    # Sort by improvement (descending)
    organ_trend_data.sort(key=lambda x: x[3], reverse=True)
    
    organs_trend = [item[0] for item in organ_trend_data[:10]]  # Top 10
    first_losses = [item[1] for item in organ_trend_data[:10]]
    last_losses = [item[2] for item in organ_trend_data[:10]]
    
    x = np.arange(len(organs_trend))
    width = 0.35
    
    plt.bar(x - width/2, first_losses, width, label='First 10 epochs avg', alpha=0.7)
    plt.bar(x + width/2, last_losses, width, label='Last 10 epochs avg', alpha=0.7)
    
    plt.title('Loss Improvement: First vs Last 10 Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Organs')
    plt.ylabel('Average Loss')
    plt.xticks(x, organs_trend, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, active_organs, top_organs

def save_data_summary(overall_avg, organ_avg, epochs, active_organs, output_dir):
    """Save extracted data to CSV files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save overall loss data
    overall_df = pd.DataFrame({
        'Epoch': epochs,
        'Overall_Loss': [overall_avg[epoch] for epoch in epochs]
    })
    overall_df.to_csv(output_dir / 'overall_loss.csv', index=False)
    
    # Save organ loss data
    organ_data = {'Epoch': epochs}
    for organ in active_organs:
        organ_data[f'{organ}_loss'] = [organ_avg[organ][epoch] for epoch in epochs]
    
    organ_df = pd.DataFrame(organ_data)
    organ_df.to_csv(output_dir / 'organ_losses.csv', index=False)
    
    print(f"Data saved to {output_dir}")

def main():
    # File paths
    log_file = "/home/muhammedg/fvlm/training_20250826_125919.log"
    output_dir = "/home/muhammedg/fvlm/loss_analysis"
    
    # Parse the log file
    epoch_data, overall_loss_data, organs = parse_training_log(log_file)
    
    # Calculate averages
    overall_avg, organ_avg, epochs = calculate_epoch_averages(epoch_data, overall_loss_data, organs)
    
    # Create plots
    fig, active_organs, top_organs = plot_loss_curves(overall_avg, organ_avg, epochs, organs)
    
    # Save plots
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plot_file = output_path / 'loss_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Loss curves plot saved to: {plot_file}")
    
    # Save data
    save_data_summary(overall_avg, organ_avg, epochs, active_organs, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING LOSS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total epochs analyzed: {len(epochs)}")
    print(f"Total organs tracked: {len(organs)}")
    print(f"Active organs (with non-zero losses): {len(active_organs)}")
    print(f"\nActive organs: {', '.join(active_organs)}")
    print(f"\nTop organs by maximum loss: {', '.join(top_organs[:5])}")
    
    # Overall loss trend
    start_loss = overall_avg[epochs[0]]
    end_loss = overall_avg[epochs[-1]]
    print(f"\nOverall loss trend:")
    print(f"  Start (epoch {epochs[0]}): {start_loss:.4f}")
    print(f"  End (epoch {epochs[-1]}): {end_loss:.4f}")
    print(f"  Improvement: {start_loss - end_loss:.4f} ({((start_loss - end_loss) / start_loss * 100):.1f}%)")
    
    plt.show()

if __name__ == "__main__":
    main()
