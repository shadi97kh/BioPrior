#!/usr/bin/env python
"""
FIXED VERSION - HANDLES DIFFERENT CSV STRUCTURES
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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


def inspect_csv_structure(dataset_name):
    """
    Inspect the structure of the CSV file to understand columns
    """
    csv_path = f"./data/{dataset_name}.csv"
    if not os.path.exists(csv_path):
        csv_path = f"./data/unnorm/{dataset_name}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=5)  # Read just first 5 rows
        print(f"\n{dataset_name}.csv structure:")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print("\nFirst row example:")
        print(df.iloc[0])
        return df.columns.tolist()
    return None


def load_real_sequences(dataset_name='Mix'):
    """
    Load real siRNA sequences with better error handling
    """
    print(f"\nLoading {dataset_name} dataset...")
    
    # Try CSV first
    csv_path = f"./data/{dataset_name}.csv"
    if not os.path.exists(csv_path):
        csv_path = f"./data/unnorm/{dataset_name}.csv"
    
    if os.path.exists(csv_path):
        # Read CSV without type specification first
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} sequences from {csv_path}")
        print(f"Columns found: {df.columns.tolist()}")
        
        sequences = []
        
        for idx, row in df.iterrows():
            try:
                # Find the siRNA sequence - try different possible column names
                seq = None
                for col in ['siRNA', 'sequence', 'seq', 'Sequence', 'siRNA_seq']:
                    if col in df.columns:
                        seq = str(row[col]).upper()
                        break
                
                # If no specific column found, check if sequences are in index
                if seq is None:
                    # Maybe sequences are row indices?
                    if isinstance(row.name, str) and len(row.name) >= 19:
                        seq = row.name.upper()
                    else:
                        # Try first string column that looks like a sequence
                        for col in df.columns:
                            val = str(row[col])
                            if len(val) >= 19 and all(c in 'AUGCT' for c in val.upper()):
                                seq = val.upper()
                                break
                
                if seq is None or len(seq) < 4:
                    continue
                
                # Get label/efficacy
                label = 0.5  # default
                for col in ['label', 'y', 'efficacy', 'Label', 'Y']:
                    if col in df.columns:
                        try:
                            label = float(row[col])
                            break
                        except:
                            pass
                
                # Analyze 5' composition
                seq = seq.replace('T', 'U')  # Convert T to U for RNA
                au_count = sum(1 for nt in seq[:4] if nt in ['A', 'U'])
                
                sequences.append({
                    'sequence': seq[:19],  # Use first 19nt
                    'efficacy': label,
                    'au_count': au_count,
                    '5prime': seq[:4],
                    'category': 'AU-rich' if au_count >= 3 else 'GC-rich' if au_count <= 1 else 'Mixed'
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Successfully processed {len(sequences)} sequences")
        return sequences
    
    print(f"CSV file not found for {dataset_name}")
    return []


def create_tensor_from_sequence(sequence):
    """Create properly formatted tensor"""
    tensor = torch.ones(1, 1, 19, 5).to(device)
    
    nucleotide_map = {'A': 1, 'U': 2, 'T': 2, 'G': 3, 'C': 4}
    for i, nt in enumerate(sequence[:19]):
        if nt.upper() in nucleotide_map:
            tensor[0, 0, i, nucleotide_map[nt.upper()]] = 2.0
    
    tensor.requires_grad = True
    return tensor


def analyze_dataset_simple(dataset_name='Mix'):
    """
    Simplified analysis focusing on key metrics
    """
    print("="*60)
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print("="*60)
    
    # First inspect the CSV structure
    inspect_csv_structure(dataset_name)
    
    # Load sequences
    sequences = load_real_sequences(dataset_name)
    
    if not sequences:
        print(f"No sequences loaded from {dataset_name}")
        return None
    
    # Statistics
    au_rich = [s for s in sequences if s['au_count'] >= 3]
    gc_rich = [s for s in sequences if s['au_count'] <= 1]
    mixed = [s for s in sequences if 1 < s['au_count'] < 3]
    
    print(f"\n5' Composition Distribution:")
    print(f"  A/U-rich (3-4 A/U): {len(au_rich)} ({100*len(au_rich)/len(sequences):.1f}%)")
    print(f"  G/C-rich (0-1 A/U): {len(gc_rich)} ({100*len(gc_rich)/len(sequences):.1f}%)")
    print(f"  Mixed (2 A/U): {len(mixed)} ({100*len(mixed)/len(sequences):.1f}%)")
    
    # If we have efficacy labels, analyze them
    if sequences[0]['efficacy'] != 0.5:
        au_effs = [s['efficacy'] for s in au_rich if s['efficacy'] != 0.5]
        gc_effs = [s['efficacy'] for s in gc_rich if s['efficacy'] != 0.5]
        
        if au_effs and gc_effs:
            print(f"\nEfficacy by 5' Composition:")
            print(f"  A/U-rich: {np.mean(au_effs):.3f} ± {np.std(au_effs):.3f}")
            print(f"  G/C-rich: {np.mean(gc_effs):.3f} ± {np.std(gc_effs):.3f}")
    
    # Load model for gradient analysis
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
    
    # Sample sequences for gradient analysis
    sample_size = min(10, len(au_rich), len(gc_rich))
    
    if sample_size > 0:
        print(f"\nGradient Analysis (sample of {sample_size} sequences per category):")
        
        results = []
        
        for category, seq_list in [('A/U-rich', au_rich[:sample_size]), 
                                   ('G/C-rich', gc_rich[:sample_size])]:
            
            grads = []
            for seq_data in seq_list:
                seq = seq_data['sequence']
                
                # Create inputs
                siRNA = create_tensor_from_sequence(seq)
                mRNA = torch.ones(1, 1, 57, 5).to(device)
                siRNA_FM = torch.randn(1, 256, 19, 5).to(device) * 0.1
                mRNA_FM = torch.randn(1, 256, 57, 5).to(device) * 0.1
                td = torch.zeros(1, 24).to(device)
                
                td[0, 0] = seq_data['au_count'] / 4.0
                
                # Forward and backward
                output, _, _ = model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
                efficacy = output[0, 1]
                
                model.zero_grad()
                efficacy.backward()
                
                if siRNA.grad is not None:
                    grad_5prime = siRNA.grad[0, 0, :4, :].abs()
                    grads.append(grad_5prime.mean().item())
            
            if grads:
                print(f"  {category}: {np.mean(grads):.6f} ± {np.std(grads):.6f}")
                results.extend(grads)
        
        if len(results) >= 2*sample_size:
            au_grads = results[:sample_size]
            gc_grads = results[sample_size:]
            if np.mean(gc_grads) > 0:
                print(f"  Gradient ratio (A/U : G/C): {np.mean(au_grads)/np.mean(gc_grads):.2f}")
    
    return sequences


def main():
    """
    Main analysis with better error handling
    """
    all_results = {}
    
    for dataset in ['Hu', 'Mix', 'Taka']:
        try:
            sequences = analyze_dataset_simple(dataset)
            if sequences:
                all_results[dataset] = sequences
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save what we got
    if all_results:
        os.makedirs("./results/saliency", exist_ok=True)
        
        # Save summary
        summary = {}
        for dataset, sequences in all_results.items():
            au_rich = [s for s in sequences if s['au_count'] >= 3]
            gc_rich = [s for s in sequences if s['au_count'] <= 1]
            
            summary[dataset] = {
                'total_sequences': len(sequences),
                'au_rich_count': len(au_rich),
                'gc_rich_count': len(gc_rich),
                'au_rich_percentage': 100*len(au_rich)/len(sequences) if sequences else 0,
                'gc_rich_percentage': 100*len(gc_rich)/len(sequences) if sequences else 0
            }
        
        with open("./results/saliency/dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to ./results/saliency/dataset_summary.json")
        
        # Create simple visualization
        if len(all_results) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            datasets = list(summary.keys())
            au_percentages = [summary[d]['au_rich_percentage'] for d in datasets]
            gc_percentages = [summary[d]['gc_rich_percentage'] for d in datasets]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            ax.bar(x - width/2, au_percentages, width, label='A/U-rich (≥3)', color='green')
            ax.bar(x + width/2, gc_percentages, width, label='G/C-rich (≤1)', color='red')
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Percentage of Sequences')
            ax.set_title('5\' Composition Distribution Across Datasets')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('./results/saliency/dataset_distribution.png')
            print("Plot saved to ./results/saliency/dataset_distribution.png")


if __name__ == "__main__":
    main()