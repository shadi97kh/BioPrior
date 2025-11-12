#!/usr/bin/env python
"""
REALISTIC siRNA COMPARISON ANALYSIS
Compares naturally occurring similar siRNAs to understand what drives efficacy differences
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import mannwhitneyu, ttest_ind
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_dataset(dataset_name='Hu'):
    """Load and prepare dataset"""
    csv_path = f'./data/{dataset_name}.csv'
    
    if not os.path.exists(csv_path):
        print(f"Dataset {dataset_name} not found")
        return None
    
    print(f"Loading dataset: {dataset_name}")
    df = pd.read_csv(csv_path)
    
    # Clean and standardize sequences
    df['siRNA_clean'] = df['siRNA'].str.upper().str.replace('T', 'U')
    df['seq_length'] = df['siRNA_clean'].str.len()
    
    # Filter to 19-mers only
    df = df[df['seq_length'] == 19].copy()
    
    # Calculate sequence features
    df['5prime'] = df['siRNA_clean'].str[:4]
    df['seed'] = df['siRNA_clean'].str[1:8]  # Positions 2-8
    df['3prime'] = df['siRNA_clean'].str[-4:]
    
    # Calculate compositional features
    for region, cols in [('5p', slice(0, 4)), ('seed', slice(1, 8)), 
                         ('central', slice(8, 13)), ('3p', slice(15, 19))]:
        df[f'AU_{region}'] = df['siRNA_clean'].str[cols].apply(
            lambda x: sum(1 for nt in x if nt in 'AU') / len(x))
        df[f'GC_{region}'] = df['siRNA_clean'].str[cols].apply(
            lambda x: sum(1 for nt in x if nt in 'GC') / len(x))
    
    print(f"Processed {len(df)} sequences")
    return df


def find_natural_similar_pairs(df, max_distance=2, min_efficacy_diff=0.3):
    """Find naturally occurring similar sequences with different efficacies"""
    
    print(f"\nFinding similar sequence pairs in real data...")
    print(f"Criteria: ≤{max_distance} mutations, ≥{min_efficacy_diff} efficacy difference")
    
    sequences = df['siRNA_clean'].values
    efficacies = df['label'].values
    
    pairs = []
    
    # Use vectorized comparison for efficiency
    n = len(sequences)
    
    # Sample if dataset is too large
    if n > 500:
        sample_indices = np.random.choice(n, 500, replace=False)
    else:
        sample_indices = range(n)
    
    for i in tqdm(sample_indices, desc="Finding pairs"):
        seq1 = sequences[i]
        eff1 = efficacies[i]
        
        for j in range(i + 1, n):
            seq2 = sequences[j]
            eff2 = efficacies[j]
            
            # Quick efficacy difference check first (faster)
            eff_diff = abs(eff1 - eff2)
            if eff_diff < min_efficacy_diff:
                continue
            
            # Count differences
            differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
            
            if differences <= max_distance and differences > 0:
                # Analyze the differences
                diff_positions = []
                diff_details = []
                
                for pos, (nt1, nt2) in enumerate(zip(seq1, seq2)):
                    if nt1 != nt2:
                        region = 'seed' if 1 <= pos <= 7 else ('5prime' if pos < 4 else 
                                ('central' if pos < 13 else '3prime'))
                        
                        diff_details.append({
                            'position': pos + 1,
                            'from': nt1,
                            'to': nt2,
                            'region': region,
                            'mutation': f"{nt1}{pos+1}{nt2}"
                        })
                        diff_positions.append(pos + 1)
                
                pairs.append({
                    'seq1': seq1,
                    'seq2': seq2,
                    'efficacy1': eff1,
                    'efficacy2': eff2,
                    'efficacy_diff': eff_diff,
                    'efficacy_change': eff2 - eff1,  # Directional change
                    'n_differences': differences,
                    'diff_positions': diff_positions,
                    'diff_details': diff_details,
                    'higher_efficacy_seq': seq2 if eff2 > eff1 else seq1
                })
    
    pairs_df = pd.DataFrame(pairs)
    
    if len(pairs_df) == 0:
        print("No suitable pairs found! Try relaxing criteria.")
        return None
    
    # Sort by efficacy difference
    pairs_df = pairs_df.sort_values('efficacy_diff', ascending=False)
    
    print(f"Found {len(pairs_df)} similar sequence pairs")
    print(f"Average efficacy difference: {pairs_df['efficacy_diff'].mean():.3f}")
    
    return pairs_df


def analyze_mutation_patterns(pairs_df):
    """Analyze which mutations correlate with efficacy changes"""
    
    print("\nAnalyzing mutation patterns...")
    
    # Track mutations and their effects
    mutation_effects = {}
    position_effects = {i: [] for i in range(1, 20)}
    region_effects = {'5prime': [], 'seed': [], 'central': [], '3prime': []}
    
    # Specific mutation types
    mutation_types = {
        'AU_to_GC': [],  # Weakening mutations
        'GC_to_AU': [],  # Strengthening mutations  
        'purine_swap': [],  # A<->G
        'pyrimidine_swap': [],  # U<->C
        'transition': [],  # Purine to purine or pyrimidine to pyrimidine
        'transversion': []  # Purine to pyrimidine or vice versa
    }
    
    for _, pair in pairs_df.iterrows():
        eff_change = pair['efficacy_change']
        
        for diff in pair['diff_details']:
            pos = diff['position']
            from_nt = diff['from']
            to_nt = diff['to']
            region = diff['region']
            
            # Normalize effect by number of differences
            normalized_effect = eff_change / pair['n_differences']
            
            # Track position and region effects
            position_effects[pos].append(normalized_effect)
            region_effects[region].append(normalized_effect)
            
            # Track specific mutations
            mutation_key = f"{from_nt}->{to_nt}"
            if mutation_key not in mutation_effects:
                mutation_effects[mutation_key] = []
            mutation_effects[mutation_key].append(normalized_effect)
            
            # Classify mutation type
            if from_nt in 'AU' and to_nt in 'GC':
                mutation_types['AU_to_GC'].append(normalized_effect)
            elif from_nt in 'GC' and to_nt in 'AU':
                mutation_types['GC_to_AU'].append(normalized_effect)
            
            if (from_nt in 'AG' and to_nt in 'AG'):
                mutation_types['purine_swap'].append(normalized_effect)
            elif (from_nt in 'UC' and to_nt in 'UC'):
                mutation_types['pyrimidine_swap'].append(normalized_effect)
            
            purines = 'AG'
            pyrimidines = 'UC'
            if (from_nt in purines and to_nt in purines) or \
               (from_nt in pyrimidines and to_nt in pyrimidines):
                mutation_types['transition'].append(normalized_effect)
            else:
                mutation_types['transversion'].append(normalized_effect)
    
    # Calculate statistics
    results = {
        'position_stats': {},
        'mutation_stats': {},
        'region_stats': {},
        'mutation_type_stats': {}
    }
    
    # Position statistics
    for pos, effects in position_effects.items():
        if effects:
            results['position_stats'][pos] = {
                'mean': np.mean(effects),
                'std': np.std(effects),
                'median': np.median(effects),
                'count': len(effects),
                'abs_mean': np.mean(np.abs(effects))
            }
    
    # Mutation statistics
    for mut, effects in mutation_effects.items():
        if len(effects) >= 3:  # Need minimum samples
            results['mutation_stats'][mut] = {
                'mean': np.mean(effects),
                'std': np.std(effects),
                'median': np.median(effects),
                'count': len(effects)
            }
    
    # Region statistics
    for region, effects in region_effects.items():
        if effects:
            results['region_stats'][region] = {
                'mean': np.mean(effects),
                'std': np.std(effects),
                'median': np.median(effects),
                'count': len(effects),
                'abs_mean': np.mean(np.abs(effects))
            }
    
    # Mutation type statistics
    for mut_type, effects in mutation_types.items():
        if effects:
            results['mutation_type_stats'][mut_type] = {
                'mean': np.mean(effects),
                'std': np.std(effects),
                'count': len(effects)
            }
    
    return results


def analyze_5prime_patterns(df):
    """Specifically analyze 5' region patterns in real sequences"""
    
    print("\nAnalyzing 5' region patterns in real data...")
    
    # Get unique 5' sequences and their average efficacies
    fiveprime_stats = df.groupby('5prime').agg({
        'label': ['mean', 'std', 'count']
    }).reset_index()
    fiveprime_stats.columns = ['5prime', 'mean_efficacy', 'std_efficacy', 'count']
    
    # Filter for sequences with enough samples
    fiveprime_stats = fiveprime_stats[fiveprime_stats['count'] >= 5]
    
    # Calculate AU content for each 5' sequence
    fiveprime_stats['AU_content'] = fiveprime_stats['5prime'].apply(
        lambda x: sum(1 for nt in x if nt in 'AU') / 4
    )
    
    # Sort by mean efficacy
    fiveprime_stats = fiveprime_stats.sort_values('mean_efficacy', ascending=False)
    
    print(f"Found {len(fiveprime_stats)} unique 5' sequences with ≥5 samples")
    
    # Identify patterns
    top10 = fiveprime_stats.head(10)
    bottom10 = fiveprime_stats.tail(10)
    
    print("\nTop 10 5' sequences by efficacy:")
    for _, row in top10.iterrows():
        print(f"  {row['5prime']}: {row['mean_efficacy']:.3f} ± {row['std_efficacy']:.3f} "
              f"(n={row['count']}, AU={row['AU_content']:.1%})")
    
    print("\nBottom 10 5' sequences by efficacy:")
    for _, row in bottom10.iterrows():
        print(f"  {row['5prime']}: {row['mean_efficacy']:.3f} ± {row['std_efficacy']:.3f} "
              f"(n={row['count']}, AU={row['AU_content']:.1%})")
    
    return fiveprime_stats


def create_comprehensive_visualization(pairs_df, mutation_results, fiveprime_stats):
    """Create detailed visualization of findings"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Position-wise mutation effects
    ax1 = fig.add_subplot(gs[0, :2])
    positions = sorted(mutation_results['position_stats'].keys())
    mean_effects = [mutation_results['position_stats'][p]['mean'] 
                   for p in positions]
    abs_effects = [mutation_results['position_stats'][p]['abs_mean'] 
                  for p in positions]
    
    x = np.array(positions)
    ax1.bar(x, mean_effects, color=['green' if e > 0 else 'red' for e in mean_effects],
            alpha=0.7, label='Mean effect')
    ax1.plot(x, abs_effects, 'k--', marker='o', label='Absolute mean', alpha=0.5)
    
    ax1.set_xlabel('Position in siRNA')
    ax1.set_ylabel('Mean Efficacy Change')
    ax1.set_title('Positional Impact of Mutations (Real Data)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add region shading
    ax1.axvspan(0.5, 4.5, alpha=0.1, color='blue', label='5\' region')
    ax1.axvspan(1.5, 8.5, alpha=0.1, color='orange', label='Seed')
    
    # 2. Mutation type effects
    ax2 = fig.add_subplot(gs[0, 2])
    mut_types = mutation_results['mutation_type_stats']
    if mut_types:
        types = list(mut_types.keys())
        means = [mut_types[t]['mean'] for t in types]
        counts = [mut_types[t]['count'] for t in types]
        
        colors = ['green' if m > 0 else 'red' for m in means]
        bars = ax2.barh(types, means, color=colors, alpha=0.7)
        
        # Add count annotations
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' n={count}', ha='left' if width > 0 else 'right',
                    va='center', fontsize=9)
        
        ax2.set_xlabel('Mean Efficacy Change')
        ax2.set_title('Mutation Type Effects')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Specific nucleotide mutations
    ax3 = fig.add_subplot(gs[0, 3])
    mut_effects = mutation_results['mutation_stats']
    if mut_effects:
        # Sort by effect size
        sorted_muts = sorted(mut_effects.items(), 
                           key=lambda x: abs(x[1]['mean']), 
                           reverse=True)[:12]
        
        muts = [m[0] for m in sorted_muts]
        means = [m[1]['mean'] for m in sorted_muts]
        counts = [m[1]['count'] for m in sorted_muts]
        
        colors = ['green' if m > 0 else 'red' for m in means]
        bars = ax3.barh(muts, means, color=colors, alpha=0.7)
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f' {count}', ha='left' if width > 0 else 'right',
                    va='center', fontsize=8)
        
        ax3.set_xlabel('Mean Efficacy Change')
        ax3.set_title('Top Nucleotide Mutations')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. 5' sequence efficacy distribution
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Plot top and bottom 5' sequences
    top_n = 15
    combined = pd.concat([
        fiveprime_stats.head(top_n).assign(group='Top'),
        fiveprime_stats.tail(top_n).assign(group='Bottom')
    ])
    
    x_pos = np.arange(len(combined))
    colors = ['green' if g == 'Top' else 'red' for g in combined['group']]
    
    bars = ax4.bar(x_pos, combined['mean_efficacy'], yerr=combined['std_efficacy'],
                   color=colors, alpha=0.7, capsize=3)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(combined['5prime'], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Mean Efficacy')
    ax4.set_title(f'Top and Bottom 5\' Sequences (Real Data)')
    ax4.axhline(y=combined['mean_efficacy'].mean(), color='black', 
               linestyle='--', alpha=0.5, label='Overall mean')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. AU content correlation
    ax5 = fig.add_subplot(gs[1, 2])
    
    scatter = ax5.scatter(fiveprime_stats['AU_content'] * 4,  # Convert to count
                         fiveprime_stats['mean_efficacy'],
                         s=fiveprime_stats['count'] * 5,  # Size by sample count
                         alpha=0.6, c=fiveprime_stats['AU_content'],
                         cmap='coolwarm')
    
    # Add trend line
    z = np.polyfit(fiveprime_stats['AU_content'] * 4, 
                   fiveprime_stats['mean_efficacy'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 4, 100)
    ax5.plot(x_line, p(x_line), "r--", alpha=0.8, 
            label=f'Slope={z[0]:.3f}')
    
    ax5.set_xlabel('A/U Count in 5\' Region')
    ax5.set_ylabel('Mean Efficacy')
    ax5.set_title('5\' A/U Content vs Efficacy (Real Sequences)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='A/U Fraction')
    
    # 6. Example similar pairs with different efficacy
    ax6 = fig.add_subplot(gs[2:, :2])
    ax6.axis('off')
    
    # Show top pairs
    top_pairs = pairs_df.head(8)
    
    example_text = "SIMILAR SEQUENCES WITH DIFFERENT EFFICACY\n" + "="*50 + "\n\n"
    
    for idx, (_, pair) in enumerate(top_pairs.iterrows(), 1):
        example_text += f"Pair {idx}: ΔEfficacy = {pair['efficacy_diff']:.3f}\n"
        
        # Highlight differences
        seq1 = pair['seq1']
        seq2 = pair['seq2']
        
        # Create aligned display
        display1 = ""
        display2 = ""
        markers = ""
        
        for i, (a, b) in enumerate(zip(seq1, seq2)):
            if a != b:
                display1 += f"[{a}]"
                display2 += f"[{b}]"
                markers += " ▼ "
            else:
                display1 += f" {a} "
                display2 += f" {b} "
                markers += "   "
        
        example_text += f"  E={pair['efficacy1']:.2f}: {display1}\n"
        example_text += f"         {markers}\n"
        example_text += f"  E={pair['efficacy2']:.2f}: {display2}\n"
        
        # Show mutations
        mutations = [d['mutation'] for d in pair['diff_details']]
        example_text += f"  Changes: {', '.join(mutations)}\n\n"
    
    ax6.text(0.05, 0.95, example_text, transform=ax6.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 7. Statistical summary
    ax7 = fig.add_subplot(gs[2:, 2:])
    ax7.axis('off')
    
    summary = "STATISTICAL SUMMARY\n" + "="*40 + "\n\n"
    
    # Overall statistics
    summary += f"Similar pairs analyzed: {len(pairs_df)}\n"
    summary += f"Average efficacy difference: {pairs_df['efficacy_diff'].mean():.3f}\n"
    summary += f"Max efficacy difference: {pairs_df['efficacy_diff'].max():.3f}\n\n"
    
    # Region importance
    summary += "REGIONAL IMPACT:\n"
    for region, stats in mutation_results['region_stats'].items():
        summary += f"  {region:8s}: {stats['mean']:+.4f} (n={stats['count']})\n"
    
    # Most impactful positions
    summary += "\nMOST IMPACTFUL POSITIONS:\n"
    sorted_pos = sorted(mutation_results['position_stats'].items(),
                       key=lambda x: x[1]['abs_mean'],
                       reverse=True)[:5]
    for pos, stats in sorted_pos:
        summary += f"  Position {pos:2d}: {stats['mean']:+.4f} (n={stats['count']})\n"
    
    # Best 5' sequences
    summary += "\nBEST 5' SEQUENCES (REAL DATA):\n"
    for _, row in fiveprime_stats.head(5).iterrows():
        summary += f"  {row['5prime']}: {row['mean_efficacy']:.3f} "
        summary += f"(AU={int(row['AU_content']*4)}/4, n={int(row['count'])})\n"
    
    # Statistical tests
    summary += "\nSTATISTICAL TESTS:\n"
    
    # Test AU-rich vs GC-rich 5' sequences
    au_rich = fiveprime_stats[fiveprime_stats['AU_content'] >= 0.75]['mean_efficacy']
    gc_rich = fiveprime_stats[fiveprime_stats['AU_content'] <= 0.25]['mean_efficacy']
    
    if len(au_rich) > 0 and len(gc_rich) > 0:
        t_stat, p_val = ttest_ind(au_rich, gc_rich)
        summary += f"  AU-rich vs GC-rich 5': p={p_val:.3e}\n"
        summary += f"    AU-rich mean: {au_rich.mean():.3f}\n"
        summary += f"    GC-rich mean: {gc_rich.mean():.3f}\n"
    
    ax7.text(0.05, 0.95, summary, transform=ax7.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Realistic siRNA Comparison Analysis - Natural Sequence Variations', 
                fontsize=14, fontweight='bold')
    
    return fig


def main():
    """Main analysis pipeline"""
    
    print("="*70)
    print("REALISTIC siRNA COMPARISON - NATURAL SEQUENCES")
    print("="*70)
    
    # Analyze each dataset
    all_pairs = []
    all_fiveprime_stats = []
    
    for dataset_name in ['Hu', 'Taka']:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print('='*50)
        
        # Load data
        df = load_dataset(dataset_name)
        if df is None:
            continue
        
        # Find similar pairs
        pairs_df = find_natural_similar_pairs(df, 
                                             max_distance=2, 
                                             min_efficacy_diff=0.2)
        
        if pairs_df is not None:
            pairs_df['dataset'] = dataset_name
            all_pairs.append(pairs_df)
        
        # Analyze 5' patterns
        fiveprime_stats = analyze_5prime_patterns(df)
        fiveprime_stats['dataset'] = dataset_name
        all_fiveprime_stats.append(fiveprime_stats)
    
    if not all_pairs:
        print("\nNo suitable pairs found!")
        return
    
    # Combine results
    combined_pairs = pd.concat(all_pairs, ignore_index=True)
    combined_fiveprime = pd.concat(all_fiveprime_stats, ignore_index=True)
    
    print(f"\n{'='*50}")
    print(f"Combined Analysis")
    print('='*50)
    print(f"Total pairs: {len(combined_pairs)}")
    print(f"Unique 5' sequences: {len(combined_fiveprime)}")
    
    # Analyze mutation patterns
    mutation_results = analyze_mutation_patterns(combined_pairs)
    
    # Create visualization
    fig = create_comprehensive_visualization(combined_pairs, 
                                            mutation_results, 
                                            combined_fiveprime)
    
    # Save results
    os.makedirs('./results/realistic_comparison', exist_ok=True)
    
    fig.savefig('./results/realistic_comparison/analysis.png', 
               dpi=200, bbox_inches='tight')
    print("\n✅ Saved visualization")
    
    combined_pairs.to_csv('./results/realistic_comparison/similar_pairs.csv', 
                         index=False)
    combined_fiveprime.to_csv('./results/realistic_comparison/5prime_stats.csv', 
                             index=False)
    print("✅ Saved data files")
    
    plt.show()


if __name__ == "__main__":
    main()