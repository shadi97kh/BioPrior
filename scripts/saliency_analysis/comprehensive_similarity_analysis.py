#!/usr/bin/env python
"""
COMPREHENSIVE siRNA SIMILARITY ANALYSIS
Processes all datasets and finds similar sequences with adaptive criteria
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr
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
        print(f"Dataset {dataset_name} not found at {csv_path}")
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


def find_similar_pairs_adaptive(df, dataset_name):
    """Find similar sequences with adaptive criteria"""
    
    print(f"\nFinding similar sequence pairs in {dataset_name}...")
    
    sequences = df['siRNA_clean'].values
    efficacies = df['label'].values
    
    # Try different criteria levels
    criteria_levels = [
        {'max_distance': 1, 'min_efficacy_diff': 0.3, 'target_pairs': 50},
        {'max_distance': 2, 'min_efficacy_diff': 0.2, 'target_pairs': 50},
        {'max_distance': 3, 'min_efficacy_diff': 0.15, 'target_pairs': 50},
        {'max_distance': 4, 'min_efficacy_diff': 0.1, 'target_pairs': 50},
    ]
    
    all_pairs = []
    
    for criteria in criteria_levels:
        print(f"  Trying: ≤{criteria['max_distance']} mutations, "
              f"≥{criteria['min_efficacy_diff']} efficacy difference")
        
        pairs = []
        n = len(sequences)
        
        # For large datasets, use sampling
        if n > 1000:
            # Sample more comprehensively
            sample_size = min(1000, n)
            sample_indices = np.random.choice(n, sample_size, replace=False)
        else:
            sample_indices = range(n)
        
        # Find pairs
        for i in tqdm(sample_indices, desc="  Scanning", leave=False):
            if len(pairs) >= criteria['target_pairs']:
                break
                
            seq1 = sequences[i]
            eff1 = efficacies[i]
            
            # Compare with all other sequences (not just subsequent ones)
            compare_indices = np.random.choice(n, min(100, n), replace=False)
            
            for j in compare_indices:
                if i == j:
                    continue
                    
                seq2 = sequences[j]
                eff2 = efficacies[j]
                
                # Check efficacy difference
                eff_diff = abs(eff1 - eff2)
                if eff_diff < criteria['min_efficacy_diff']:
                    continue
                
                # Count differences
                differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
                
                if differences <= criteria['max_distance'] and differences > 0:
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
                        'efficacy_change': eff2 - eff1,
                        'n_differences': differences,
                        'diff_positions': diff_positions,
                        'diff_details': diff_details,
                        'criteria_level': criteria_levels.index(criteria)
                    })
                    
                    if len(pairs) >= criteria['target_pairs']:
                        break
        
        print(f"    Found {len(pairs)} pairs")
        all_pairs.extend(pairs)
        
        # If we have enough pairs, stop
        if len(all_pairs) >= 50:
            break
    
    if not all_pairs:
        print(f"  No pairs found for {dataset_name}")
        return None
    
    pairs_df = pd.DataFrame(all_pairs)
    
    # Remove duplicate pairs (same sequences in different order)
    pairs_df['seq_pair'] = pairs_df.apply(
        lambda x: tuple(sorted([x['seq1'], x['seq2']])), axis=1)
    pairs_df = pairs_df.drop_duplicates(subset=['seq_pair'])
    
    # Sort by efficacy difference
    pairs_df = pairs_df.sort_values('efficacy_diff', ascending=False)
    
    # Take top pairs
    pairs_df = pairs_df.head(100)
    
    print(f"  Final: {len(pairs_df)} unique pairs")
    print(f"  Avg efficacy diff: {pairs_df['efficacy_diff'].mean():.3f}")
    print(f"  Avg mutations: {pairs_df['n_differences'].mean():.1f}")
    
    return pairs_df


def analyze_mutation_patterns(pairs_df):
    """Analyze which mutations correlate with efficacy changes"""
    
    # Track mutations and their effects
    mutation_effects = {}
    position_effects = {i: [] for i in range(1, 20)}
    region_effects = {'5prime': [], 'seed': [], 'central': [], '3prime': []}
    
    # Specific mutation types
    mutation_types = {
        'AU_to_GC': [],
        'GC_to_AU': [],
        'purine_swap': [],
        'pyrimidine_swap': [],
        'transition': [],
        'transversion': []
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
            
            if (from_nt in 'AG' and to_nt in 'AG') or \
               (from_nt in 'UC' and to_nt in 'UC'):
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
        if len(effects) >= 2:  # Lower threshold for more results
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
    """Analyze 5' region patterns in real sequences"""
    
    # Get unique 5' sequences and their average efficacies
    fiveprime_stats = df.groupby('5prime').agg({
        'label': ['mean', 'std', 'count']
    }).reset_index()
    fiveprime_stats.columns = ['5prime', 'mean_efficacy', 'std_efficacy', 'count']
    
    # Filter for sequences with enough samples (lower threshold for more data)
    fiveprime_stats = fiveprime_stats[fiveprime_stats['count'] >= 3]
    
    # Calculate AU content
    fiveprime_stats['AU_content'] = fiveprime_stats['5prime'].apply(
        lambda x: sum(1 for nt in x if nt in 'AU') / 4
    )
    
    # Sort by mean efficacy
    fiveprime_stats = fiveprime_stats.sort_values('mean_efficacy', ascending=False)
    
    return fiveprime_stats


def create_combined_visualization(all_results):
    """Create comprehensive visualization for all datasets"""
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    # Combine all pairs for overall analysis
    all_pairs = pd.concat([r['pairs'] for r in all_results if r['pairs'] is not None], 
                          ignore_index=True)
    all_fiveprime = pd.concat([r['fiveprime_stats'] for r in all_results], 
                              ignore_index=True)
    
    if len(all_pairs) > 0:
        combined_mutation_results = analyze_mutation_patterns(all_pairs)
    else:
        combined_mutation_results = None
    
    # 1. Position-wise mutation effects across all datasets
    ax1 = fig.add_subplot(gs[0, :3])
    
    if combined_mutation_results:
        positions = sorted(combined_mutation_results['position_stats'].keys())
        mean_effects = [combined_mutation_results['position_stats'][p]['mean'] 
                       for p in positions]
        counts = [combined_mutation_results['position_stats'][p]['count'] 
                 for p in positions]
        
        colors = ['green' if e > 0 else 'red' for e in mean_effects]
        bars = ax1.bar(positions, mean_effects, color=colors, alpha=0.7)
        
        # Add count annotations
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=7)
        
        ax1.set_xlabel('Position in siRNA')
        ax1.set_ylabel('Mean Efficacy Change')
        ax1.set_title('Positional Impact of Mutations (All Datasets Combined)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add region shading
        ax1.axvspan(0.5, 4.5, alpha=0.1, color='blue')
        ax1.axvspan(1.5, 8.5, alpha=0.1, color='orange')
        ax1.text(2.5, ax1.get_ylim()[1]*0.9, '5\'', ha='center', fontsize=9)
        ax1.text(5, ax1.get_ylim()[1]*0.9, 'Seed', ha='center', fontsize=9)
    
    # 2. Dataset comparison
    ax2 = fig.add_subplot(gs[0, 3:])
    dataset_names = [r['dataset'] for r in all_results]
    pair_counts = [len(r['pairs']) if r['pairs'] is not None else 0 
                   for r in all_results]
    colors_dataset = ['blue', 'orange', 'green'][:len(dataset_names)]
    
    bars = ax2.bar(dataset_names, pair_counts, color=colors_dataset, alpha=0.7)
    ax2.set_ylabel('Number of Similar Pairs Found')
    ax2.set_title('Dataset Comparison')
    
    for bar, count in zip(bars, pair_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    # 3. Top 5' sequences across all datasets
    ax3 = fig.add_subplot(gs[1, :3])
    
    # Get top and bottom sequences
    top_n = 12
    top_seqs = all_fiveprime.nlargest(top_n, 'mean_efficacy')
    bottom_seqs = all_fiveprime.nsmallest(top_n, 'mean_efficacy')
    combined = pd.concat([
        top_seqs.assign(group='Top'),
        bottom_seqs.assign(group='Bottom')
    ])
    
    x_pos = np.arange(len(combined))
    colors = ['green' if g == 'Top' else 'red' for g in combined['group']]
    
    bars = ax3.bar(x_pos, combined['mean_efficacy'], 
                   yerr=combined['std_efficacy'],
                   color=colors, alpha=0.7, capsize=2, error_kw={'linewidth': 0.5})
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{seq}\n({ds})" for seq, ds in 
                         zip(combined['5prime'], combined['dataset'])], 
                        rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Mean Efficacy')
    ax3.set_title('Best and Worst 5\' Sequences Across All Datasets')
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. AU content correlation
    ax4 = fig.add_subplot(gs[1, 3])
    
    # Color by dataset
    dataset_colors = {'Hu': 'blue', 'Taka': 'orange', 'Mix': 'green'}
    colors = [dataset_colors.get(d, 'gray') for d in all_fiveprime['dataset']]
    
    scatter = ax4.scatter(all_fiveprime['AU_content'] * 4,
                         all_fiveprime['mean_efficacy'],
                         s=all_fiveprime['count'] * 3,
                         alpha=0.6, c=colors)
    
    # Add trend line
    z = np.polyfit(all_fiveprime['AU_content'] * 4, 
                   all_fiveprime['mean_efficacy'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 4, 100)
    ax4.plot(x_line, p(x_line), "r--", alpha=0.8, 
            label=f'Slope={z[0]:.3f}')
    
    ax4.set_xlabel('A/U Count in 5\' Region')
    ax4.set_ylabel('Mean Efficacy')
    ax4.set_title('5\' A/U Content vs Efficacy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add dataset legend
    for dataset, color in dataset_colors.items():
        ax4.scatter([], [], c=color, alpha=0.6, s=50, label=dataset)
    ax4.legend(loc='lower right', fontsize=8)
    
    # 5. Mutation type effects
    ax5 = fig.add_subplot(gs[1, 4])
    
    if combined_mutation_results and combined_mutation_results['mutation_type_stats']:
        mut_types = combined_mutation_results['mutation_type_stats']
        types = list(mut_types.keys())
        means = [mut_types[t]['mean'] for t in types]
        counts = [mut_types[t]['count'] for t in types]
        
        colors = ['green' if m > 0 else 'red' for m in means]
        bars = ax5.barh(types, means, color=colors, alpha=0.7)
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2,
                    f' {count}', ha='left' if width > 0 else 'right',
                    va='center', fontsize=8)
        
        ax5.set_xlabel('Mean Efficacy Change')
        ax5.set_title('Mutation Type Effects')
        ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 6. Example pairs from each dataset
    ax6 = fig.add_subplot(gs[2:, :3])
    ax6.axis('off')
    
    example_text = "EXAMPLE SIMILAR SEQUENCES BY DATASET\n" + "="*60 + "\n\n"
    
    for result in all_results:
        if result['pairs'] is None or len(result['pairs']) == 0:
            continue
            
        example_text += f"Dataset: {result['dataset']}\n"
        example_text += "-"*30 + "\n"
        
        # Show top 3 pairs
        top_pairs = result['pairs'].head(3)
        
        for idx, (_, pair) in enumerate(top_pairs.iterrows(), 1):
            example_text += f"Pair {idx}: ΔE={pair['efficacy_diff']:.3f}, "
            example_text += f"{pair['n_differences']} mutations\n"
            
            # Show sequences with differences highlighted
            seq1 = pair['seq1']
            seq2 = pair['seq2']
            
            display1 = ""
            display2 = ""
            for i, (a, b) in enumerate(zip(seq1, seq2)):
                if a != b:
                    display1 += f"[{a}]"
                    display2 += f"[{b}]"
                else:
                    display1 += a
                    display2 += b
            
            example_text += f"  {display1} (E={pair['efficacy1']:.2f})\n"
            example_text += f"  {display2} (E={pair['efficacy2']:.2f})\n"
            
            # Show mutations
            mutations = [d['mutation'] for d in pair['diff_details']]
            example_text += f"  Changes: {', '.join(mutations)}\n\n"
        
        example_text += "\n"
    
    ax6.text(0.02, 0.98, example_text, transform=ax6.transAxes,
            fontsize=7, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 7. Summary statistics
    ax7 = fig.add_subplot(gs[2:, 3:])
    ax7.axis('off')
    
    summary = "OVERALL SUMMARY\n" + "="*40 + "\n\n"
    
    # Dataset statistics
    summary += "DATASET STATISTICS:\n"
    for result in all_results:
        summary += f"  {result['dataset']:6s}: "
        if result['pairs'] is not None:
            summary += f"{len(result['pairs'])} pairs, "
            summary += f"avg ΔE={result['pairs']['efficacy_diff'].mean():.3f}\n"
        else:
            summary += "No pairs found\n"
    
    summary += f"\nTotal pairs analyzed: {len(all_pairs)}\n"
    
    if len(all_pairs) > 0:
        summary += f"Average efficacy difference: {all_pairs['efficacy_diff'].mean():.3f}\n"
        summary += f"Average mutations: {all_pairs['n_differences'].mean():.1f}\n\n"
        
        # Regional impact
        if combined_mutation_results:
            summary += "REGIONAL IMPACT:\n"
            for region, stats in combined_mutation_results['region_stats'].items():
                summary += f"  {region:8s}: {stats['mean']:+.5f} (n={stats['count']})\n"
            
            # Most impactful positions
            summary += "\nMOST IMPACTFUL POSITIONS:\n"
            sorted_pos = sorted(combined_mutation_results['position_stats'].items(),
                               key=lambda x: x[1]['abs_mean'] if x[1]['count'] > 0 else 0,
                               reverse=True)[:5]
            for pos, stats in sorted_pos:
                if stats['count'] > 0:
                    summary += f"  Position {pos:2d}: {stats['mean']:+.5f} (n={stats['count']})\n"
    
    # Best 5' sequences
    summary += "\nBEST 5' SEQUENCES (ALL DATA):\n"
    for _, row in all_fiveprime.head(5).iterrows():
        summary += f"  {row['5prime']}: {row['mean_efficacy']:.3f} "
        summary += f"(AU={int(row['AU_content']*4)}/4, n={int(row['count'])}, {row['dataset']})\n"
    
    # Statistical test
    au_rich = all_fiveprime[all_fiveprime['AU_content'] >= 0.75]['mean_efficacy']
    gc_rich = all_fiveprime[all_fiveprime['AU_content'] <= 0.25]['mean_efficacy']
    
    if len(au_rich) > 0 and len(gc_rich) > 0:
        t_stat, p_val = ttest_ind(au_rich, gc_rich)
        summary += f"\nSTATISTICAL TEST:\n"
        summary += f"  AU-rich (≥3/4) vs GC-rich (≤1/4):\n"
        summary += f"    p-value: {p_val:.3e}\n"
        summary += f"    AU-rich mean: {au_rich.mean():.3f} (n={len(au_rich)})\n"
        summary += f"    GC-rich mean: {gc_rich.mean():.3f} (n={len(gc_rich)})\n"
        summary += f"    Difference: {au_rich.mean() - gc_rich.mean():.3f}\n"
    
    ax7.text(0.02, 0.98, summary, transform=ax7.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Comprehensive siRNA Similarity Analysis - All Datasets', 
                fontsize=14, fontweight='bold')
    
    return fig


def main():
    """Main analysis pipeline"""
    
    print("="*70)
    print("COMPREHENSIVE siRNA SIMILARITY ANALYSIS")
    print("="*70)
    
    # Get all available datasets
    if os.path.exists('./data/'):
        available_datasets = [f.replace('.csv', '') for f in os.listdir('./data/') 
                             if f.endswith('.csv')]
        print(f"\nAvailable datasets: {available_datasets}")
    else:
        print("No data directory found!")
        return
    
    # Process ALL datasets
    all_results = []
    
    for dataset_name in available_datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print('='*50)
        
        # Load data
        df = load_dataset(dataset_name)
        if df is None:
            continue
        
        # Find similar pairs with adaptive criteria
        pairs_df = find_similar_pairs_adaptive(df, dataset_name)
        
        # Analyze 5' patterns
        print(f"\nAnalyzing 5' patterns in {dataset_name}...")
        fiveprime_stats = analyze_5prime_patterns(df)
        fiveprime_stats['dataset'] = dataset_name
        
        print(f"Found {len(fiveprime_stats)} unique 5' sequences with ≥3 samples")
        
        # Show top sequences
        if len(fiveprime_stats) > 0:
            print("\nTop 5 sequences by efficacy:")
            for _, row in fiveprime_stats.head(5).iterrows():
                print(f"  {row['5prime']}: {row['mean_efficacy']:.3f} ± {row['std_efficacy']:.3f} "
                      f"(n={int(row['count'])}, AU={row['AU_content']:.1%})")
        
        # Store results
        result = {
            'dataset': dataset_name,
            'df': df,
            'pairs': pairs_df,
            'fiveprime_stats': fiveprime_stats
        }
        
        if pairs_df is not None:
            result['mutation_results'] = analyze_mutation_patterns(pairs_df)
        else:
            result['mutation_results'] = None
        
        all_results.append(result)
    
    # Create combined visualization
    print(f"\n{'='*50}")
    print("Creating combined visualization...")
    print('='*50)
    
    fig = create_combined_visualization(all_results)
    
    # Save results
    os.makedirs('./results/comprehensive_similarity', exist_ok=True)
    
    fig.savefig('./results/comprehensive_similarity/analysis.png', 
               dpi=200, bbox_inches='tight')
    print("\n✅ Saved visualization to ./results/comprehensive_similarity/analysis.png")
    
    # Save data
    all_pairs = []
    all_fiveprime = []
    
    for result in all_results:
        if result['pairs'] is not None:
            pairs = result['pairs'].copy()
            pairs['dataset'] = result['dataset']
            all_pairs.append(pairs)
        
        fiveprime = result['fiveprime_stats'].copy()
        all_fiveprime.append(fiveprime)
    
    if all_pairs:
        combined_pairs = pd.concat(all_pairs, ignore_index=True)
        combined_pairs.to_csv('./results/comprehensive_similarity/all_pairs.csv', 
                             index=False)
        print(f"✅ Saved {len(combined_pairs)} pairs to all_pairs.csv")
    
    if all_fiveprime:
        combined_fiveprime = pd.concat(all_fiveprime, ignore_index=True)
        combined_fiveprime.to_csv('./results/comprehensive_similarity/5prime_stats.csv', 
                                 index=False)
        print(f"✅ Saved {len(combined_fiveprime)} 5' sequences to 5prime_stats.csv")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['dataset']}:")
        print(f"  Sequences analyzed: {len(result['df'])}")
        print(f"  Similar pairs found: {len(result['pairs']) if result['pairs'] is not None else 0}")
        print(f"  Unique 5' patterns: {len(result['fiveprime_stats'])}")
        
        if result['pairs'] is not None and len(result['pairs']) > 0:
            print(f"  Avg efficacy diff in pairs: {result['pairs']['efficacy_diff'].mean():.3f}")
            print(f"  Avg mutations in pairs: {result['pairs']['n_differences'].mean():.1f}")
    
    plt.show()


if __name__ == "__main__":
    main()
