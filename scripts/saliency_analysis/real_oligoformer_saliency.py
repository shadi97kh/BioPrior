#!/usr/bin/env python
"""
FINAL CORRECTED VERSION - PROPER GRADIENT COMPUTATION

THE GRADIENT PROBLEM:
When you modify tensor.data after creating with requires_grad=True,
it breaks the gradient graph. The tensor is no longer a leaf tensor.

THE SOLUTION:
Create the tensor with values FIRST, then set requires_grad=True
OR use retain_grad() to force gradient storage
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import siRNA_Encoder, mRNA_Encoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FixedOligo(nn.Module):
    """Fixed OligoFormer with correct dimensions"""
    
    def __init__(self):
        super().__init__()
        # Original encoders
        self.siRNA_encoder = siRNA_Encoder(26, 128, 32, 8, 1)
        self.mRNA_encoder = mRNA_Encoder(26, 128, 32, 8, 1, 19, 19)
        
        # Pooling
        self.siRNA_avgpool = nn.AvgPool2d((19, 5))
        self.mRNA_avgpool = nn.AvgPool2d((57, 5))
        
        # Corrected classifier: 5144 input features
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
        # Encode
        siRNA, siRNA_attention = self.siRNA_encoder(siRNA)
        mRNA, mRNA_attention = self.mRNA_encoder(mRNA)
        
        # Pool RNA-FM
        siRNA_FM = self.siRNA_avgpool(siRNA_FM)
        mRNA_FM = self.mRNA_avgpool(mRNA_FM)
        
        # Fix view dimensions
        siRNA_FM = siRNA_FM.view(siRNA_FM.shape[0], -1)
        mRNA_FM = mRNA_FM.view(mRNA_FM.shape[0], -1)
        
        # Flatten
        siRNA = self.flatten(siRNA)
        mRNA = self.flatten(mRNA)
        siRNA_FM = self.flatten(siRNA_FM)
        mRNA_FM = self.flatten(mRNA_FM)
        td = self.flatten(td)
        
        # Merge and classify
        merge = torch.cat([siRNA, mRNA, siRNA_FM, mRNA_FM, td], dim=-1)
        x = self.classifier(merge)
        
        return x, siRNA_attention, mRNA_attention


def create_sirna_tensor(sequence):
    """
    CREATE TENSOR PROPERLY FOR GRADIENT COMPUTATION
    
    Key: Build the tensor with values FIRST, then set requires_grad
    """
    # Create tensor WITHOUT requires_grad first
    tensor = torch.ones(1, 1, 19, 5).to(device)
    
    # Set values
    nucleotide_map = {'A': 1, 'U': 2, 'T': 2, 'G': 3, 'C': 4}
    for i, nt in enumerate(sequence[:19]):
        if nt in nucleotide_map:
            tensor[0, 0, i, nucleotide_map[nt]] = 2.0
    
    # NOW set requires_grad
    tensor.requires_grad = True
    
    return tensor


def analyze_5prime():
    """Main analysis with proper gradient handling"""
    
    print("="*60)
    print("5' A/U GRADIENT ANALYSIS - CORRECTED")
    print("="*60)
    
    # Load model
    model = FixedOligo().to(device)
    
    model_path = "./model/best_model.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        saved_state = torch.load(model_path, map_location=device)
        
        # Load encoder weights only
        model_dict = model.state_dict()
        loaded = 0
        for name, param in saved_state.items():
            if name in model_dict and 'classifier' not in name:
                if param.shape == model_dict[name].shape:
                    model_dict[name] = param
                    loaded += 1
        
        model.load_state_dict(model_dict)
        print(f"Loaded {loaded} encoder weights")
    
    # CRITICAL: Set to training mode
    model.train()
    print("Model in training mode\n")
    
    # Test sequences
    sequences = [
        ('AAUU', 'AAUUCGCGAUCGAUCGAUC'),
        ('AUAU', 'AUAUCGCGAUCGAUCGAUC'),
        ('UAUA', 'UAUACGCGAUCGAUCGAUC'),
        ('UUAA', 'UUAACGCGAUCGAUCGAUC'),
        ('GGCC', 'GGCCCGCGAUCGAUCGAUC'),
        ('GCGC', 'GCGCCGCGAUCGAUCGAUC'),
        ('CGCG', 'CGCGCGCGAUCGAUCGAUC'),
        ('CCGG', 'CCGGCGCGAUCGAUCGAUC')
    ]
    
    print("Computing gradients:")
    print("-" * 40)
    
    results = []
    
    for label, seq in sequences:
        # CREATE TENSORS PROPERLY
        siRNA = create_sirna_tensor(seq)  # This now has requires_grad=True
        
        # Other inputs (don't need gradients)
        mRNA = torch.ones(1, 1, 57, 5).to(device)
        siRNA_FM = torch.randn(1, 256, 19, 5).to(device) * 0.1
        mRNA_FM = torch.randn(1, 256, 57, 5).to(device) * 0.1
        td = torch.zeros(1, 24).to(device)
        
        # Set 5' composition features
        au_count = sum(1 for nt in seq[:4] if nt in ['A', 'U', 'T'])
        td[0, 0] = au_count / 4.0
        td[0, 1] = (4 - au_count) / 4.0
        
        # Forward
        output, _, _ = model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
        efficacy = output[0, 1]
        
        # Backward
        model.zero_grad()
        efficacy.backward()
        
        # Now gradients should exist
        if siRNA.grad is not None:
            grad_5prime = siRNA.grad[0, 0, :4, :].abs()
            mean_grad = grad_5prime.mean().item()
            pos_grads = grad_5prime.mean(dim=1).cpu().numpy()
        else:
            print(f"  WARNING: No gradient for {label}")
            mean_grad = 0
            pos_grads = np.zeros(4)
        
        results.append({
            '5prime': label,
            'sequence': seq,
            'efficacy': efficacy.item(),
            'mean_gradient': mean_grad,
            'position_gradients': pos_grads.tolist(),
            'au_count': au_count
        })
        
        print(f"  {label}: Eff={efficacy.item():.4f}, Grad={mean_grad:.6f}, A/U={au_count}/4")
    
    # Analysis
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    au_rich = [r for r in results if r['au_count'] >= 3]
    gc_rich = [r for r in results if r['au_count'] <= 1]
    
    if au_rich and gc_rich:
        au_grads = [r['mean_gradient'] for r in au_rich]
        gc_grads = [r['mean_gradient'] for r in gc_rich]
        au_effs = [r['efficacy'] for r in au_rich]
        gc_effs = [r['efficacy'] for r in gc_rich]
        
        print(f"\nGRADIENTS:")
        print(f"  A/U-rich: {np.mean(au_grads):.6f}")
        print(f"  G/C-rich: {np.mean(gc_grads):.6f}")
        if np.mean(gc_grads) > 0:
            print(f"  Ratio: {np.mean(au_grads)/np.mean(gc_grads):.2f}")
        
        print(f"\nEFFICACY:")
        print(f"  A/U-rich: {np.mean(au_effs):.4f}")
        print(f"  G/C-rich: {np.mean(gc_effs):.4f}")
        
        print(f"\nPOSITION-WISE (A/U-rich):")
        pos_data = np.array([r['position_gradients'] for r in au_rich])
        for i in range(4):
            print(f"  Position {i+1}: {pos_data[:, i].mean():.6f}")
    
    # Save
    os.makedirs("./results/saliency", exist_ok=True)
    with open("./results/saliency/gradient_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to ./results/saliency/gradient_results.json")
    
    # Plot
    plot_results(results)
    
    return results


def plot_results(results):
    """Create visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    au_rich = [r for r in results if r['au_count'] >= 3]
    gc_rich = [r for r in results if r['au_count'] <= 1]
    
    # 1. Gradients
    ax = axes[0]
    au_grads = [r['mean_gradient'] for r in au_rich]
    gc_grads = [r['mean_gradient'] for r in gc_rich]
    
    bp = ax.boxplot([au_grads, gc_grads], labels=['A/U-rich', 'G/C-rich'],
                     patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('5\' Gradient Analysis')
    ax.grid(True, alpha=0.3)
    
    # 2. Position-wise
    ax = axes[1]
    pos_data = np.array([r['position_gradients'] for r in au_rich])
    means = pos_data.mean(axis=0)
    stds = pos_data.std(axis=0)
    
    x = np.arange(1, 5)
    ax.bar(x, means, yerr=stds, capsize=5, color='green', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Gradient')
    ax.set_title('Position-wise (A/U-rich)')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    
    # 3. Efficacy
    ax = axes[2]
    au_effs = [r['efficacy'] for r in au_rich]
    gc_effs = [r['efficacy'] for r in gc_rich]
    
    bp = ax.boxplot([au_effs, gc_effs], labels=['A/U-rich', 'G/C-rich'],
                     patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax.set_ylabel('Predicted Efficacy')
    ax.set_title('Model Predictions')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('OligoFormer 5\' A/U Preference Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('./results/saliency/gradient_analysis.png', dpi=150)
    print("Plot saved to ./results/saliency/gradient_analysis.png")
    plt.show()


if __name__ == "__main__":
    results = analyze_5prime()