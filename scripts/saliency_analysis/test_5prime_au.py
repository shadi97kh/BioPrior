#!/usr/bin/env python
"""
5' A/U Preference Saliency Analysis for OligoFormer
Author: [Your Name]
Date: [Date]
"""

import sys
import os
# Add parent directory to path to import OligoFormer modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import OligoFormer modules (adjust based on actual structure)
try:
    from model import OligoFormer  # Assuming model.py exists
    from utils import load_model   # Assuming utils.py exists
except ImportError:
    print("Warning: Could not import OligoFormer modules")
    print("Using dummy model for testing")

# Your test function here
def test_5prime_au_preference(model_path=None):
    """
    Minimal test to validate 5' A/U preference in your model
    """
    
    # Test sequences
    test_sequences = {
        'au_rich': ['AAUUCGCGAUCGAUCGAUC',  # A/U at positions 1-4
                    'AUAUCGCGAUCGAUCGAUC',
                    'UAUACGCGAUCGAUCGAUC',
                    'UUAACGCGAUCGAUCGAUC'],
        'gc_rich': ['GGCCCGCGAUCGAUCGAUC',  # G/C at positions 1-4
                    'GCGCCGCGAUCGAUCGAUC',
                    'CGCGCGCGAUCGAUCGAUC',
                    'CCGGCGCGAUCGAUCGAUC']
    }
    
    # Convert to one-hot encoding
    def sequence_to_onehot(seq):
        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        onehot = torch.zeros(1, len(seq), 4)
        for i, nt in enumerate(seq):
            onehot[0, i, mapping[nt]] = 1
        return onehot
    
    # Load model
    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}")
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using dummy model instead")
            model = DummyModel()
    else:
        print("No model path provided or file not found. Using dummy model.")
        
        # For demonstration, using a dummy model
        class DummyModel(nn.Module):
            def forward(self, x):
                # Simulate preference for A/U at 5' end
                au_score = x[0, :4, :2].sum()  # A and U channels
                gc_score = x[0, :4, 2:].sum()  # G and C channels
                return (au_score - gc_score).unsqueeze(0)
        
        model = DummyModel()
    
    # Compute attributions
    ig = IntegratedGradients(model)
    
    results = {'au_rich': [], 'gc_rich': []}
    
    for seq_type, sequences in test_sequences.items():
        for seq in sequences:
            input_tensor = sequence_to_onehot(seq)
            baseline = torch.zeros_like(input_tensor)
            
            # Compute attribution
            attribution = ig.attribute(input_tensor, baseline, n_steps=50)
            
            # Focus on first 4 positions
            five_prime_attr = attribution[0, :4, :].sum(dim=1).numpy()
            
            results[seq_type].append({
                'sequence': seq,
                '5prime': seq[:4],
                'attributions': five_prime_attr,
                'mean_attribution': five_prime_attr.mean()
            })
    
    # Print results
    print("\n" + "="*60)
    print("5' A/U Preference Analysis Results:")
    print("="*60)
    
    au_mean = np.mean([r['mean_attribution'] for r in results['au_rich']])
    gc_mean = np.mean([r['mean_attribution'] for r in results['gc_rich']])
    
    print(f"A/U-rich 5' mean attribution: {au_mean:.4f}")
    print(f"G/C-rich 5' mean attribution: {gc_mean:.4f}")
    print(f"Ratio (A/U : G/C): {au_mean/gc_mean:.2f}")
    
    # Save results
    save_results(results)
    
    # Create visualization
    visualize_results(results)
    
    return results

def save_results(results, output_dir="../../results/saliency/attribution_scores"):
    """Save attribution results to file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    import json
    with open(f"{output_dir}/5prime_au_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value_list in results.items():
            json_results[key] = []
            for item in value_list:
                json_item = item.copy()
                json_item['attributions'] = item['attributions'].tolist()
                json_results[key].append(json_item)
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/5prime_au_results.json")

def visualize_results(results, output_dir="../../results/saliency/figures"):
    """Create and save visualization"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Attribution comparison
    ax = axes[0]
    au_attrs = [r['mean_attribution'] for r in results['au_rich']]
    gc_attrs = [r['mean_attribution'] for r in results['gc_rich']]
    
    positions = [1, 2]
    means = [np.mean(au_attrs), np.mean(gc_attrs)]
    stds = [np.std(au_attrs), np.std(gc_attrs)]
    
    bars = ax.bar(positions, means, yerr=stds, capsize=10,
                   color=['green', 'red'], alpha=0.7,
                   tick_label=['A/U-rich 5\'', 'G/C-rich 5\''])
    ax.set_ylabel('Mean Attribution Score')
    ax.set_title('5\' Composition Impact on Model Attention')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Position-wise attribution for A/U-rich
    ax = axes[1]
    au_positions = np.array([r['attributions'] for r in results['au_rich']])
    mean_positions = au_positions.mean(axis=0)
    std_positions = au_positions.std(axis=0)
    
    positions = np.arange(1, 5)
    ax.bar(positions, mean_positions, yerr=std_positions, capsize=5, color='green', alpha=0.7)
    ax.set_xlabel('Position at 5\' End')
    ax.set_ylabel('Attribution Score')
    ax.set_title('Position-wise Attribution (A/U-rich sequences)')
    ax.set_xticks(positions)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/5prime_au_attribution.png", dpi=300)
    plt.show()
    print(f"Figure saved to {output_dir}/5prime_au_attribution.png")

def main():
    """Main execution"""
    # Set paths relative to script location
    script_dir = Path(__file__).parent
    model_path = script_dir / "../../pretrained/oligoformer_model.pth"
    
    print("Starting 5' A/U Preference Analysis")
    print(f"Looking for model at: {model_path}")
    
    # Run analysis
    results = test_5prime_au_preference(str(model_path))
    
    return results

if __name__ == "__main__":
    results = main()