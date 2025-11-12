#!/usr/bin/env python
"""
TRUE SALIENCY MAP VISUALIZATION FOR OLIGOFORMER
Creates actual saliency heatmaps showing importance of each nucleotide position
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from matplotlib.patches import Rectangle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import siRNA_Encoder, mRNA_Encoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FixedOligo(nn.Module):
    """Fixed OligoFormer"""
    def __init__(self):
        super().__init__()
        self.siRNA_encoder = siRNA_Encoder(26, 128, 32, 8, 1)
        self.mRNA_encoder = mRNA_Encoder(26, 128, 32, 8, 1, 19, 19)
        self.siRNA_avgpool = nn.AvgPool2d((19, 5))
        self.mRNA_avgpool = nn.AvgPool2d((57, 5))
        self.classifier = nn.Sequential(
            nn.Linear(5144, 256),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.flatten = nn.Flatten()
    
    def forward(self, siRNA, mRNA, siRNA_FM, mRNA_FM, td):
        siRNA, siRNA_attention = self.siRNA_encoder(siRNA)
        mRNA, mRNA_attention = self.mRNA_encoder(mRNA)
        siRNA_FM = self.siRNA_avgpool(siRNA_FM)
        mRNA_FM = self.mRNA_avgpool(mRNA_FM)
        siRNA_FM = siRNA_FM.view(siRNA_FM.shape[0], -1)
        mRNA_FM = mRNA_FM.view(mRNA_FM.shape[0], -1)
        siRNA = self.flatten(siRNA)
        mRNA = self.flatten(mRNA)
        siRNA_FM = self.flatten(siRNA_FM)
        mRNA_FM = self.flatten(mRNA_FM)
        td = self.flatten(td)
        merge = torch.cat([siRNA, mRNA, siRNA_FM, mRNA_FM, td], dim=-1)
        x = self.classifier(merge)
        return x, siRNA_attention, mRNA_attention


def create_input_tensor(sequence):
    """Create input tensor from sequence"""
    tensor = torch.ones(1, 1, 19, 5).to(device)
    nucleotide_map = {'A': 1, 'U': 2, 'T': 2, 'G': 3, 'C': 4}
    for i, nt in enumerate(sequence[:19]):
        if nt.upper() in nucleotide_map:
            tensor[0, 0, i, nucleotide_map[nt.upper()]] = 2.0
    tensor.requires_grad = True
    return tensor


def compute_saliency_map(model, sequence):
    """
    Compute saliency map for a single sequence
    Returns gradient magnitude for each position and nucleotide
    """
    # Create inputs
    siRNA = create_input_tensor(sequence)
    mRNA = torch.ones(1, 1, 57, 5).to(device)
    siRNA_FM = torch.randn(1, 256, 19, 5).to(device) * 0.1
    mRNA_FM = torch.randn(1, 256, 57, 5).to(device) * 0.1
    td = torch.zeros(1, 24).to(device)
    
    # Set 5' features
    au_count = sum(1 for nt in sequence[:4] if nt.upper() in ['A', 'U', 'T'])
    td[0, 0] = au_count / 4.0
    
    # Forward pass
    output, _, _ = model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
    efficacy = output[0, 1]
    
    # Backward pass
    model.zero_grad()
    efficacy.backward()
    
    # Get gradient map (19 positions x 5 channels)
    # Channel 0 is constant, channels 1-4 are A,U,G,C
    gradient_map = siRNA.grad[0, 0, :, :].abs().cpu().numpy()
    
    return gradient_map, efficacy.item()


def create_saliency_visualization():
    """
    Create comprehensive saliency map visualization
    """
    print("="*60)
    print("GENERATING SALIENCY MAPS")
    print("="*60)
    
    # Load model
    model = FixedOligo().to(device)
    model_path = "./model/best_model.pth"
    if os.path.exists(model_path):
        saved_state = torch.load(model_path, map_location=device)
        model_dict = model.state_dict()
        for name, param in saved_state.items():
            if name in model_dict and 'classifier' not in name:
                if param.shape == model_dict[name].shape:
                    model_dict[name] = param
        model.load_state_dict(model_dict)
    model.train()
    
    # Test sequences from your results
    test_sequences = {
        'A/U-rich': [
            ('CUAAUAUGUUAAUUGAUUU', 'Example from Hu dataset'),
            ('AAUUCGCGAUCGAUCGAUC', 'Synthetic A/U-rich'),
            ('UUAACGCGAUCGAUCGAUC', 'Synthetic A/U-rich'),
            ('AUAUCGCGAUCGAUCGAUC', 'Synthetic A/U-rich'),
        ],
        'G/C-rich': [
            ('UUCUCUGGAAUGCCUGCAC', 'Example from Mix dataset'),
            ('GGCCCGCGAUCGAUCGAUC', 'Synthetic G/C-rich'),
            ('CCGGCGCGAUCGAUCGAUC', 'Synthetic G/C-rich'),
            ('GCGCCGCGAUCGAUCGAUC', 'Synthetic G/C-rich'),
        ]
    }
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Individual saliency maps
    print("\nComputing individual saliency maps...")
    for group_idx, (group_name, sequences) in enumerate(test_sequences.items()):
        for seq_idx, (seq, desc) in enumerate(sequences[:2]):  # Show 2 examples per group
            ax = plt.subplot(2, 4, group_idx*4 + seq_idx + 1)
            
            # Compute saliency
            gradient_map, efficacy = compute_saliency_map(model, seq)
            
            # Extract nucleotide-specific gradients (channels 1-4)
            nucleotide_gradients = gradient_map[:, 1:5]  # Shape: (19, 4)
            
            # Create heatmap
            im = ax.imshow(nucleotide_gradients.T, cmap='Reds', aspect='auto')
            
            # Formatting
            ax.set_title(f'{group_name}\n{seq[:8]}...\nEff: {efficacy:.3f}', fontsize=10)
            ax.set_xlabel('Position')
            ax.set_ylabel('Nucleotide')
            ax.set_xticks([0, 4, 9, 14, 18])
            ax.set_xticklabels([1, 5, 10, 15, 19])
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['A', 'U', 'G', 'C'])
            
            # Add 5' region highlight
            rect = Rectangle((0, -0.5), 4, 4, linewidth=2, 
                           edgecolor='blue', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            
            # Add colorbar for first plot
            if seq_idx == 0 and group_idx == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 2: Average saliency map by category
    print("\nComputing average saliency maps...")
    ax_avg_au = plt.subplot(2, 4, 5)
    ax_avg_gc = plt.subplot(2, 4, 6)
    
    for ax, (group_name, sequences) in zip([ax_avg_au, ax_avg_gc], test_sequences.items()):
        all_maps = []
        for seq, _ in sequences:
            gradient_map, _ = compute_saliency_map(model, seq)
            all_maps.append(gradient_map[:, 1:5])
        
        # Average across sequences
        avg_map = np.mean(all_maps, axis=0)
        
        im = ax.imshow(avg_map.T, cmap='Reds', aspect='auto')
        ax.set_title(f'Average {group_name} Saliency', fontsize=12)
        ax.set_xlabel('Position')
        ax.set_ylabel('Nucleotide')
        ax.set_xticks([0, 4, 9, 14, 18])
        ax.set_xticklabels([1, 5, 10, 15, 19])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['A', 'U', 'G', 'C'])
        
        # Highlight 5' region
        rect = Rectangle((0, -0.5), 4, 4, linewidth=2, 
                       edgecolor='blue', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 3: Position-wise importance (summed across nucleotides)
    ax_pos = plt.subplot(2, 4, 7)
    
    au_positions = []
    gc_positions = []
    
    for group_name, sequences in test_sequences.items():
        for seq, _ in sequences:
            gradient_map, _ = compute_saliency_map(model, seq)
            pos_importance = gradient_map[:, 1:5].sum(axis=1)  # Sum across nucleotides
            
            if group_name == 'A/U-rich':
                au_positions.append(pos_importance)
            else:
                gc_positions.append(pos_importance)
    
    au_mean = np.mean(au_positions, axis=0)
    gc_mean = np.mean(gc_positions, axis=0)
    
    x = np.arange(1, 20)
    ax_pos.plot(x, au_mean, 'g-', label='A/U-rich', linewidth=2)
    ax_pos.plot(x, gc_mean, 'r-', label='G/C-rich', linewidth=2)
    ax_pos.axvspan(1, 4, alpha=0.2, color='blue', label='5\' region')
    ax_pos.set_xlabel('Position')
    ax_pos.set_ylabel('Total Gradient Magnitude')
    ax_pos.set_title('Position-wise Importance')
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3)
    
    # Plot 4: 5' region focus
    ax_5prime = plt.subplot(2, 4, 8)
    
    # Create detailed 5' heatmap
    au_5prime_maps = []
    gc_5prime_maps = []
    
    for group_name, sequences in test_sequences.items():
        for seq, _ in sequences:
            gradient_map, _ = compute_saliency_map(model, seq)
            five_prime_map = gradient_map[:4, 1:5]  # First 4 positions, 4 nucleotides
            
            if group_name == 'A/U-rich':
                au_5prime_maps.append(five_prime_map)
            else:
                gc_5prime_maps.append(five_prime_map)
    
    # Calculate difference
    au_avg_5prime = np.mean(au_5prime_maps, axis=0)
    gc_avg_5prime = np.mean(gc_5prime_maps, axis=0)
    diff_map = au_avg_5prime - gc_avg_5prime
    
    im = ax_5prime.imshow(diff_map.T, cmap='RdBu_r', aspect='auto', vmin=-0.00002, vmax=0.00002)
    ax_5prime.set_title('5\' Region Difference\n(A/U-rich - G/C-rich)', fontsize=12)
    ax_5prime.set_xlabel('Position at 5\' End')
    ax_5prime.set_ylabel('Nucleotide')
    ax_5prime.set_xticks([0, 1, 2, 3])
    ax_5prime.set_xticklabels([1, 2, 3, 4])
    ax_5prime.set_yticks([0, 1, 2, 3])
    ax_5prime.set_yticklabels(['A', 'U', 'G', 'C'])
    plt.colorbar(im, ax=ax_5prime, fraction=0.046, pad=0.04)
    
    plt.suptitle('OligoFormer Saliency Maps - 5\' A/U Preference Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save
    os.makedirs('./results/saliency', exist_ok=True)
    plt.savefig('./results/saliency/saliency_maps.png', dpi=300, bbox_inches='tight')
    print("\nSaliency maps saved to ./results/saliency/saliency_maps.png")
    plt.show()
    
    return fig


def create_dataset_specific_saliency(dataset_name='Mix', num_sequences=10):
    """
    Create saliency maps for specific dataset sequences
    """
    print(f"\nGenerating saliency maps for {dataset_name} dataset...")
    
    # Load sequences from CSV
    csv_path = f"./data/{dataset_name}.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset {dataset_name} not found")
        return
    
    df = pd.read_csv(csv_path, nrows=num_sequences)
    
    # Load model
    model = FixedOligo().to(device)
    model_path = "./model/best_model.pth"
    if os.path.exists(model_path):
        saved_state = torch.load(model_path, map_location=device)
        model_dict = model.state_dict()
        for name, param in saved_state.items():
            if name in model_dict and 'classifier' not in name:
                if param.shape == model_dict[name].shape:
                    model_dict[name] = param
        model.load_state_dict(model_dict)
    model.train()
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, row in df.iterrows():
        if idx >= 10:
            break
            
        seq = str(row['siRNA']).upper().replace('T', 'U')
        label = float(row['label'])
        
        # Compute saliency
        gradient_map, efficacy = compute_saliency_map(model, seq)
        nucleotide_gradients = gradient_map[:, 1:5]
        
        # Plot
        ax = axes[idx]
        im = ax.imshow(nucleotide_gradients.T, cmap='Reds', aspect='auto')
        
        # Format
        au_count = sum(1 for nt in seq[:4] if nt in ['A', 'U'])
        ax.set_title(f'{seq[:4]}... (A/U: {au_count}/4)\nTrue: {label:.2f}, Pred: {efficacy:.2f}', 
                    fontsize=9)
        
        if idx % 5 == 0:
            ax.set_ylabel('Nucleotide')
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['A', 'U', 'G', 'C'])
        else:
            ax.set_yticks([])
        
        if idx >= 5:
            ax.set_xlabel('Position')
        ax.set_xticks([0, 4, 9, 14, 18])
        ax.set_xticklabels([1, 5, 10, 15, 19])
        
        # Highlight 5' region
        rect = Rectangle((0, -0.5), 4, 4, linewidth=1.5, 
                       edgecolor='blue', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
    
    plt.suptitle(f'{dataset_name} Dataset - Saliency Maps (First {num_sequences} Sequences)', fontsize=14)
    plt.tight_layout()
    
    output_path = f'./results/saliency/saliency_{dataset_name.lower()}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    # Create main saliency visualization
    fig = create_saliency_visualization()
    
    # Create dataset-specific visualizations
    for dataset in ['Hu', 'Mix', 'Taka']:
        create_dataset_specific_saliency(dataset, num_sequences=10)
    
    print("\nAll saliency maps generated successfully!")