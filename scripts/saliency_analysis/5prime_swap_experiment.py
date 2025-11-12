#!/usr/bin/env python
"""
COMPREHENSIVE POSITIONAL ANALYSIS FOR siRNA KNOCKDOWN EFFICACY
Identifies which positions and nucleotides contribute most to knockdown efficacy
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, f_oneway
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_dataset_with_shabalina(dataset_name):
    """Load dataset including Shabalina"""
    
    # Handle Shabalina specially
    if dataset_name == 'Shabalina':
        raw_path = '/mnt/user-data/uploads/data_2.csv'
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            
            # Standardize
            df_std = pd.DataFrame()
            df_std['siRNA'] = df['Sequence'].str.upper().str.replace('T', 'U')
            df_std['mRNA'] = df['Target seq']
            
            # Normalize activity
            activity = df['Activity']
            df_std['label'] = (activity - activity.min()) / (activity.max() - activity.min())
            df_std['y'] = (df_std['label'] > df_std['label'].median()).astype(int)
            df_std['td'] = '0,' * 23 + '0'
            
            return df_std
    else:
        # Try standard paths
        paths = [f'./data/{dataset_name}.csv', f'/data/{dataset_name}.csv']
        for path in paths:
            if os.path.exists(path):
                return pd.read_csv(path)
    
    return None


def analyze_positional_contributions(df, dataset_name):
    """Analyze which positions contribute most to efficacy"""
    
    print(f"\nAnalyzing positional contributions for {dataset_name}...")
    
    # Ensure sequences are clean
    df['siRNA_clean'] = df['siRNA'].str.upper().str.replace('T', 'U')
    
    # Filter to 19-mers
    df = df[df['siRNA_clean'].str.len() == 19].copy()
    
    if len(df) == 0:
        print(f"No valid 19-mer sequences in {dataset_name}")
        return None
    
    # Initialize results storage
    position_stats = {}
    
    # For each position (1-19)
    for pos in range(19):
        position_stats[pos + 1] = {
            'A': {'efficacies': [], 'count': 0},
            'U': {'efficacies': [], 'count': 0},
            'G': {'efficacies': [], 'count': 0},
            'C': {'efficacies': [], 'count': 0}
        }
        
        # Collect efficacies for each nucleotide at this position
        for _, row in df.iterrows():
            seq = row['siRNA_clean']
            efficacy = row['label']
            
            if pos < len(seq):
                nt = seq[pos]
                if nt in 'AUGC':
                    position_stats[pos + 1][nt]['efficacies'].append(efficacy)
                    position_stats[pos + 1][nt]['count'] += 1
    
    # Calculate statistics for each position
    position_importance = []
    
    for pos in range(1, 20):
        # Get efficacies for each nucleotide at this position
        nt_efficacies = {
            nt: position_stats[pos][nt]['efficacies'] 
            for nt in 'AUGC' 
            if position_stats[pos][nt]['efficacies']
        }
        
        if len(nt_efficacies) < 2:
            position_importance.append({
                'position': pos,
                'f_statistic': 0,
                'p_value': 1.0,
                'effect_size': 0,
                'best_nt': 'N',
                'worst_nt': 'N',
                'best_mean': 0,
                'worst_mean': 0
            })
            continue
        
        # Perform ANOVA to test if nucleotide identity matters at this position
        groups = [nt_efficacies[nt] for nt in nt_efficacies if nt_efficacies[nt]]
        
        if len(groups) >= 2:
            f_stat, p_value = f_oneway(*groups)
            
            # Calculate means for each nucleotide
            nt_means = {
                nt: np.mean(nt_efficacies[nt]) if nt in nt_efficacies else 0
                for nt in 'AUGC'
            }
            
            # Find best and worst nucleotides
            valid_nts = [(nt, mean) for nt, mean in nt_means.items() 
                        if nt in nt_efficacies and nt_efficacies[nt]]
            
            if valid_nts:
                best_nt = max(valid_nts, key=lambda x: x[1])
                worst_nt = min(valid_nts, key=lambda x: x[1])
                effect_size = best_nt[1] - worst_nt[1]
            else:
                best_nt = ('N', 0)
                worst_nt = ('N', 0)
                effect_size = 0
        else:
            f_stat = 0
            p_value = 1.0
            effect_size = 0
            best_nt = ('N', 0)
            worst_nt = ('N', 0)
        
        position_importance.append({
            'position': pos,
            'f_statistic': f_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'best_nt': best_nt[0],
            'worst_nt': worst_nt[0],
            'best_mean': best_nt[1],
            'worst_mean': worst_nt[1]
        })
    
    # Create detailed position-nucleotide matrix
    position_nt_matrix = np.zeros((4, 19))  # 4 nucleotides x 19 positions
    nt_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    
    for pos in range(1, 20):
        for nt in 'AUGC':
            if position_stats[pos][nt]['efficacies']:
                mean_efficacy = np.mean(position_stats[pos][nt]['efficacies'])
                position_nt_matrix[nt_map[nt], pos - 1] = mean_efficacy
    
    results = {
        'dataset': dataset_name,
        'position_importance': pd.DataFrame(position_importance),
        'position_stats': position_stats,
        'position_nt_matrix': position_nt_matrix,
        'n_sequences': len(df)
    }
    
    return results


def create_positional_feature_matrix(df):
    """Create one-hot encoded matrix for positional analysis"""
    
    df['siRNA_clean'] = df['siRNA'].str.upper().str.replace('T', 'U')
    df = df[df['siRNA_clean'].str.len() == 19].copy()
    
    # Create feature matrix (19 positions x 4 nucleotides = 76 features)
    features = []
    
    for _, row in df.iterrows():
        seq = row['siRNA_clean']
        seq_features = []
        
        for pos in range(19):
            # One-hot encode each position
            nt_features = [0, 0, 0, 0]
            if seq[pos] == 'A':
                nt_features[0] = 1
            elif seq[pos] in ['U', 'T']:
                nt_features[1] = 1
            elif seq[pos] == 'G':
                nt_features[2] = 1
            elif seq[pos] == 'C':
                nt_features[3] = 1
            
            seq_features.extend(nt_features)
        
        features.append(seq_features)
    
    return np.array(features), df['label'].values


def analyze_with_random_forest(df, dataset_name):
    """Use Random Forest to identify important positions"""
    
    print(f"Running Random Forest analysis for {dataset_name}...")
    
    # Create feature matrix
    X, y = create_positional_feature_matrix(df)
    
    if len(X) < 50:
        print(f"Not enough sequences for RF analysis in {dataset_name}")
        return None
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    
    # Get feature importances
    feature_importances = rf.feature_importances_
    
    # Reshape to position x nucleotide matrix
    importance_matrix = feature_importances.reshape(19, 4)
    
    # Calculate position importance (sum across nucleotides)
    position_importance = importance_matrix.sum(axis=1)
    
    # Identify most important positions
    top_positions = np.argsort(position_importance)[::-1][:10]
    
    return {
        'rf_model': rf,
        'importance_matrix': importance_matrix,
        'position_importance': position_importance,
        'top_positions': top_positions + 1,  # Convert to 1-indexed
        'r2_score': rf.score(X, y)
    }


def create_comprehensive_positional_visualization(all_results, rf_results):
    """Create detailed visualization of positional contributions"""
    
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.25)
    
    # Define dataset colors
    dataset_colors = {'Hu': 'blue', 'Taka': 'orange', 'Mix': 'green', 'Shabalina': 'red'}
    
    # 1. Position importance across all datasets (ANOVA p-values)
    ax1 = fig.add_subplot(gs[0, :])
    
    for result in all_results:
        if result is None:
            continue
        df_imp = result['position_importance']
        dataset = result['dataset']
        color = dataset_colors.get(dataset, 'gray')
        
        # Plot -log10(p-value) for each position
        positions = df_imp['position'].values
        significance = -np.log10(df_imp['p_value'].values + 1e-10)
        
        ax1.plot(positions, significance, marker='o', label=dataset, 
                color=color, alpha=0.7)
    
    ax1.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, 
               label='p=0.05 threshold')
    ax1.set_xlabel('Position in siRNA')
    ax1.set_ylabel('-log10(p-value)')
    ax1.set_title('Statistical Significance of Nucleotide Identity at Each Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 20))
    
    # Add region annotations
    ax1.axvspan(0.5, 4.5, alpha=0.1, color='blue')
    ax1.axvspan(1.5, 8.5, alpha=0.1, color='orange')
    ax1.text(2.5, ax1.get_ylim()[1]*0.95, "5'", ha='center', fontsize=9)
    ax1.text(5, ax1.get_ylim()[1]*0.95, 'Seed', ha='center', fontsize=9)
    
    # 2. Effect sizes at each position
    ax2 = fig.add_subplot(gs[1, :2])
    
    for result in all_results:
        if result is None:
            continue
        df_imp = result['position_importance']
        dataset = result['dataset']
        color = dataset_colors.get(dataset, 'gray')
        
        positions = df_imp['position'].values
        effect_sizes = df_imp['effect_size'].values
        
        ax2.bar(positions + (list(dataset_colors.keys()).index(dataset) - 1.5) * 0.2, 
               effect_sizes, width=0.2, label=dataset, color=color, alpha=0.7)
    
    ax2.set_xlabel('Position in siRNA')
    ax2.set_ylabel('Effect Size (Best - Worst NT)')
    ax2.set_title('Effect Size of Optimal vs Worst Nucleotide Choice')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(1, 20))
    
    # 3. Random Forest feature importance (if available)
    if rf_results:
        ax3 = fig.add_subplot(gs[1, 2:])
        
        for dataset, rf_result in rf_results.items():
            if rf_result is None:
                continue
            
            positions = range(1, 20)
            importance = rf_result['position_importance']
            color = dataset_colors.get(dataset, 'gray')
            
            ax3.plot(positions, importance, marker='s', label=f"{dataset} (R²={rf_result['r2_score']:.3f})", 
                    color=color, alpha=0.7)
        
        ax3.set_xlabel('Position in siRNA')
        ax3.set_ylabel('Random Forest Importance')
        ax3.set_title('Machine Learning-based Position Importance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(1, 20))
    
    # 4. Position-Nucleotide heatmaps for each dataset
    for idx, result in enumerate(all_results):
        if result is None:
            continue
        
        ax = fig.add_subplot(gs[2, idx])
        
        # Create heatmap
        matrix = result['position_nt_matrix']
        
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks(range(19))
        ax.set_xticklabels(range(1, 20), fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels(['A', 'U', 'G', 'C'])
        ax.set_xlabel('Position')
        ax.set_ylabel('Nucleotide')
        ax.set_title(f"{result['dataset']} (n={result['n_sequences']})")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Annotate with actual values for top positions
        for i in range(4):
            for j in range(19):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}', 
                                 ha="center", va="center", 
                                 color="black" if matrix[i, j] > 0.5 else "white",
                                 fontsize=5)
    
    # 5. Best nucleotide at each position (consensus)
    ax5 = fig.add_subplot(gs[3, :])
    
    # Combine all datasets
    consensus_matrix = np.zeros((4, 19))
    dataset_count = 0
    
    for result in all_results:
        if result is None:
            continue
        consensus_matrix += result['position_nt_matrix']
        dataset_count += 1
    
    if dataset_count > 0:
        consensus_matrix /= dataset_count
    
    # Find best nucleotide at each position
    best_nts = []
    nucleotides = ['A', 'U', 'G', 'C']
    
    for pos in range(19):
        best_idx = np.argmax(consensus_matrix[:, pos])
        best_nts.append(nucleotides[best_idx])
    
    # Plot as bar chart with nucleotide labels
    positions = range(1, 20)
    colors_nt = {'A': 'red', 'U': 'orange', 'G': 'green', 'C': 'blue'}
    bar_colors = [colors_nt[nt] for nt in best_nts]
    
    bars = ax5.bar(positions, [consensus_matrix[nucleotides.index(nt), i] 
                               for i, nt in enumerate(best_nts)], 
                  color=bar_colors, alpha=0.7)
    
    # Add nucleotide labels on bars
    for bar, nt in zip(bars, best_nts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                nt, ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax5.set_xlabel('Position in siRNA')
    ax5.set_ylabel('Mean Efficacy of Best Nucleotide')
    ax5.set_title('Consensus Best Nucleotide at Each Position (All Datasets)')
    ax5.set_xticks(positions)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add legend for nucleotide colors
    for nt, color in colors_nt.items():
        ax5.bar([], [], color=color, label=nt, alpha=0.7)
    ax5.legend(title='Nucleotide', loc='upper right')
    
    # 6. Top important positions summary
    ax6 = fig.add_subplot(gs[4, :2])
    ax6.axis('off')
    
    summary_text = "TOP IMPORTANT POSITIONS FOR KNOCKDOWN\n" + "="*50 + "\n\n"
    
    # Rank positions by average significance across datasets
    all_positions = {}
    for pos in range(1, 20):
        significances = []
        effect_sizes = []
        
        for result in all_results:
            if result is None:
                continue
            df_imp = result['position_importance']
            pos_data = df_imp[df_imp['position'] == pos].iloc[0]
            significances.append(-np.log10(pos_data['p_value'] + 1e-10))
            effect_sizes.append(pos_data['effect_size'])
        
        if significances:
            all_positions[pos] = {
                'avg_significance': np.mean(significances),
                'avg_effect_size': np.mean(effect_sizes),
                'consensus_best': best_nts[pos - 1]
            }
    
    # Sort by significance
    sorted_positions = sorted(all_positions.items(), 
                            key=lambda x: x[1]['avg_significance'], 
                            reverse=True)[:10]
    
    summary_text += "Position | Significance | Effect Size | Best NT\n"
    summary_text += "-" * 45 + "\n"
    
    for pos, data in sorted_positions:
        region = '5\'' if pos <= 4 else ('Seed' if pos <= 8 else 'Central' if pos <= 13 else '3\'')
        summary_text += f"   {pos:2d} ({region:7s}) |    {data['avg_significance']:5.2f}    |   {data['avg_effect_size']:5.3f}   |    {data['consensus_best']}\n"
    
    # Add design recommendations
    summary_text += "\n" + "="*50 + "\n"
    summary_text += "DESIGN RECOMMENDATIONS:\n\n"
    
    # Identify patterns
    five_prime_best = ''.join(best_nts[:4])
    seed_best = ''.join(best_nts[1:8])
    
    summary_text += f"1. Optimal 5' region: {five_prime_best}\n"
    summary_text += f"2. Optimal seed region: {seed_best}\n"
    
    # Count AU in important positions
    top_5_positions = [p[0] for p in sorted_positions[:5]]
    au_count = sum(1 for p in top_5_positions if best_nts[p-1] in ['A', 'U'])
    
    summary_text += f"3. Top 5 positions prefer {'AU' if au_count >= 3 else 'GC'} nucleotides\n"
    
    # Position-specific recommendations
    if sorted_positions[0][0] <= 4:
        summary_text += "4. Position " + str(sorted_positions[0][0]) + " in 5' region is most critical\n"
    else:
        summary_text += "4. Position " + str(sorted_positions[0][0]) + " is most critical\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 7. Statistical validation
    ax7 = fig.add_subplot(gs[4, 2:])
    ax7.axis('off')
    
    validation_text = "STATISTICAL VALIDATION\n" + "="*40 + "\n\n"
    
    # Cross-dataset consistency
    validation_text += "Cross-Dataset Consistency:\n"
    
    # Check if top positions are consistent
    top_positions_by_dataset = {}
    for result in all_results:
        if result is None:
            continue
        df_imp = result['position_importance']
        top_5 = df_imp.nsmallest(5, 'p_value')['position'].tolist()
        top_positions_by_dataset[result['dataset']] = top_5
    
    # Find common positions
    if top_positions_by_dataset:
        all_top_positions = [p for positions in top_positions_by_dataset.values() 
                            for p in positions]
        position_counts = pd.Series(all_top_positions).value_counts()
        
        validation_text += "\nPositions in top 5 across datasets:\n"
        for pos, count in position_counts.head(10).items():
            validation_text += f"  Position {pos:2d}: {count}/{len(top_positions_by_dataset)} datasets\n"
    
    # Random Forest validation
    if rf_results:
        validation_text += "\n\nRandom Forest Model Performance:\n"
        for dataset, rf_result in rf_results.items():
            if rf_result is None:
                continue
            validation_text += f"  {dataset:10s}: R² = {rf_result['r2_score']:.3f}\n"
            validation_text += f"    Top positions: {rf_result['top_positions'][:5].tolist()}\n"
    
    ax7.text(0.05, 0.95, validation_text, transform=ax7.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Comprehensive Positional Analysis for siRNA Knockdown Efficacy', 
                fontsize=16, fontweight='bold')
    
    return fig


def main():
    """Main analysis pipeline"""
    
    print("="*70)
    print("COMPREHENSIVE POSITIONAL ANALYSIS FOR siRNA EFFICACY")
    print("="*70)
    
    # Define datasets to analyze
    dataset_names = ['Hu', 'Taka', 'Mix', 'Shabalina']
    
    all_results = []
    rf_results = {}
    
    for dataset_name in dataset_names:
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name} dataset")
        print('='*50)
        
        # Load dataset
        df = load_dataset_with_shabalina(dataset_name)
        
        if df is None:
            print(f"Could not load {dataset_name}")
            all_results.append(None)
            continue
        
        print(f"Loaded {len(df)} sequences")
        
        # Analyze positional contributions
        result = analyze_positional_contributions(df, dataset_name)
        all_results.append(result)
        
        # Random Forest analysis
        rf_result = analyze_with_random_forest(df, dataset_name)
        rf_results[dataset_name] = rf_result
        
        if result:
            # Print top important positions for this dataset
            df_imp = result['position_importance']
            top_5 = df_imp.nsmallest(5, 'p_value')
            
            print(f"\nTop 5 important positions in {dataset_name}:")
            for _, row in top_5.iterrows():
                print(f"  Position {row['position']:2d}: p={row['p_value']:.4f}, "
                      f"effect={row['effect_size']:.3f}, "
                      f"best={row['best_nt']}, worst={row['worst_nt']}")
    
    # Create comprehensive visualization
    print(f"\n{'='*50}")
    print("Creating comprehensive visualization...")
    print('='*50)
    
    fig = create_comprehensive_positional_visualization(all_results, rf_results)
    
    # Save results
    os.makedirs('./results/positional_analysis', exist_ok=True)
    
    fig.savefig('./results/positional_analysis/comprehensive_positional_analysis.png', 
               dpi=200, bbox_inches='tight')
    print("\n✅ Saved visualization to ./results/positional_analysis/")
    
    # Save detailed results
    all_importance_data = []
    for result in all_results:
        if result:
            df_imp = result['position_importance'].copy()
            df_imp['dataset'] = result['dataset']
            all_importance_data.append(df_imp)
    
    if all_importance_data:
        combined_importance = pd.concat(all_importance_data, ignore_index=True)
        combined_importance.to_csv('./results/positional_analysis/position_importance.csv', 
                                  index=False)
        print("✅ Saved position importance data")
    
    # Print final summary
    print("\n" + "="*60)
    print("KEY FINDINGS - POSITIONS CRITICAL FOR KNOCKDOWN")
    print("="*60)
    
    # Aggregate across all datasets
    if all_importance_data:
        combined = pd.concat(all_importance_data)
        
        # Average by position
        avg_by_position = combined.groupby('position').agg({
            'p_value': 'mean',
            'effect_size': 'mean'
        }).sort_values('p_value')
        
        print("\nMost Important Positions (averaged across datasets):")
        print("-" * 40)
        
        for pos in avg_by_position.head(10).index:
            p_val = avg_by_position.loc[pos, 'p_value']
            effect = avg_by_position.loc[pos, 'effect_size']
            
            # Find consensus best nucleotide
            best_nts = []
            for result in all_results:
                if result:
                    pos_data = result['position_importance'][
                        result['position_importance']['position'] == pos
                    ].iloc[0]
                    best_nts.append(pos_data['best_nt'])
            
            if best_nts:
                from collections import Counter
                consensus = Counter(best_nts).most_common(1)[0][0]
            else:
                consensus = 'N'
            
            region = '5\'' if pos <= 4 else ('Seed' if pos <= 8 else 'Central' if pos <= 13 else '3\'')
            
            print(f"  Position {pos:2d} ({region:7s}): p={p_val:.4f}, "
                  f"effect={effect:.3f}, consensus best={consensus}")
    
    plt.show()
    
    return all_results, rf_results


if __name__ == "__main__":
    results, rf_results = main()