import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import argparse
import os
import sys
from collections import defaultdict

from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


# =============================================================================
#   Constants - MUST MATCH YOUR DATA ENCODING
# =============================================================================
BASES = ['A', 'U', 'G', 'C']  # FIXED: matches BioPrior and training code
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
IDX_TO_BASE = {i: b for i, b in enumerate(BASES)}


# =============================================================================
#   Physics Input Features (copied from train.py to avoid import issues)
# =============================================================================
def compute_physics_input_features(siRNA_onehot):
    """Add physics features to input."""
    x = siRNA_onehot.squeeze(1)[:, :, :4]  # [B, 19, 4]
    batch_size, seq_len, _ = x.shape
    dev = x.device
    
    seq_idx = x.argmax(dim=-1)  # [B, 19]
    
    pos_imp = torch.ones(batch_size, seq_len, 1, device=dev) / seq_len
    seed_ind = torch.zeros(batch_size, seq_len, 1, device=dev)
    seed_ind[:, 1:8, 0] = 1.0
    cleave_ind = torch.zeros(batch_size, seq_len, 1, device=dev)
    cleave_ind[:, 9:11, 0] = 1.0
    gc = ((seq_idx == 2) | (seq_idx == 3)).float().mean(dim=1, keepdim=True).unsqueeze(-1).expand(-1, seq_len, -1)
    seed_au = ((seq_idx[:, 1:8] == 0) | (seq_idx[:, 1:8] == 1)).float().mean(dim=1, keepdim=True).unsqueeze(-1).expand(-1, seq_len, -1)
    au_mask = (seq_idx == 0) | (seq_idx == 1)
    asym = (au_mask[:, :5].float().mean(dim=1) - au_mask[:, -5:].float().mean(dim=1)).view(-1, 1, 1).expand(-1, seq_len, -1)
    is_au = ((seq_idx == 0) | (seq_idx == 1)).float().unsqueeze(-1)
    is_gc = ((seq_idx == 2) | (seq_idx == 3)).float().unsqueeze(-1)
    orig = siRNA_onehot.squeeze(1)
    enhanced = torch.cat([orig, pos_imp, seed_ind, cleave_ind, gc, seed_au, asym, is_au, is_gc], dim=-1)
    return enhanced.unsqueeze(1)


# =============================================================================
#   Model Preparation (call once before loop)
# =============================================================================
def prepare_model_for_saliency(model):
    """
    Prepare model for deterministic saliency computation.
    
    CRITICAL: Order matters!
    - model.train() recursively sets ALL children to train mode
    - So we must call model.train() FIRST
    - Then override dropout modules to eval() AFTER
    
    This keeps LSTM in train mode (required for cuDNN backward)
    while disabling dropout for deterministic saliency.
    """
    # FIRST: enable train mode for cuDNN LSTM backward compatibility
    model.train()
    
    # THEN: override ALL dropout variants back to eval (deterministic)
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.eval()
    
    return model


# =============================================================================
#   Saliency Computation
# =============================================================================
def compute_saliency(model, siRNA, mRNA, siRNA_FM, mRNA_FM, td, device):
    """
    Compute gradient-based saliency for a single sample.
    Returns: [19, 4] saliency map (position x nucleotide)
    
    Note: Call prepare_model_for_saliency() once before using this function.
    """
    # Clear any accumulated gradients in model params
    model.zero_grad(set_to_none=True)
    
    # Ensure input requires grad
    siRNA_input = siRNA.clone().detach().requires_grad_(True)
    
    # Forward pass
    out = model(siRNA_input, mRNA, siRNA_FM, mRNA_FM, td)
    
    # Extract prediction
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    
    if logits.dim() == 2 and logits.size(-1) == 2:
        pred = torch.softmax(logits, dim=-1)[:, 1]
    elif logits.dim() == 1:
        pred = torch.sigmoid(logits)
    else:
        pred = logits.squeeze()
    
    # Backward to get gradients
    pred.sum().backward()
    
    # Get saliency from gradient
    grad = siRNA_input.grad  # [1, 1, 19, D] or [1, 19, D]
    
    if grad.dim() == 4:
        grad = grad.squeeze(0).squeeze(0)  # [19, D]
    elif grad.dim() == 3:
        grad = grad.squeeze(0)  # [19, D]
    
    # Take only nucleotide channels (first 4)
    saliency = grad[:, :4].abs().detach().cpu().numpy()  # [19, 4]
    
    return saliency, pred.item()


def get_position_importance(saliency):
    """
    Aggregate saliency to position importance.
    Returns: [19] array of importance scores
    """
    return saliency.sum(axis=1)  # Sum over nucleotides


def get_sequence_from_onehot(siRNA_onehot):
    if siRNA_onehot.dim() == 4:
        seq_vec = siRNA_onehot[0, 0, :, :5]  # include 5 channels
    else:
        seq_vec = siRNA_onehot[0, :, :5]

    seq = []
    for i in range(seq_vec.size(0)):
        v = seq_vec[i]
        # If unknown channel (index 4) is active or first4 sum == 0, mark as N
        if v[:4].sum().item() == 0 or v[4].item() > 0.5:
            seq.append('N')
        else:
            nt_idx = v[:4].argmax().item()
            seq.append(IDX_TO_BASE.get(nt_idx, 'N'))
    return seq


# =============================================================================
#   Perturbation Functions
# =============================================================================
def mutate_position_to_base(siRNA_onehot, position, new_base_idx):
    """
    Mutate a single position to a specific nucleotide.
    
    Args:
        siRNA_onehot: [1, 1, 19, D] or [1, 19, D] tensor
        position: int, position to mutate (0-18)
        new_base_idx: int, target nucleotide index (0-3)
    
    Returns:
        Mutated tensor (same shape)
    
    Note: Zeros ALL 5 channels to maintain valid encoding.
    Channel 4 is typically padding/unknown indicator.
    """
    mutated = siRNA_onehot.clone()
    
    if mutated.dim() == 4:
        # Zero ALL 5 channels to keep encoding valid
        mutated[0, 0, position, :5] = 0
        mutated[0, 0, position, new_base_idx] = 1
    else:
        mutated[0, position, :5] = 0
        mutated[0, position, new_base_idx] = 1
    
    return mutated


def mutate_position_random(siRNA_onehot, position):
    """Mutate a single position to a random different nucleotide."""
    if siRNA_onehot.dim() == 4:
        current_nt = siRNA_onehot[0, 0, position, :4].argmax().item()
    else:
        current_nt = siRNA_onehot[0, position, :4].argmax().item()
    
    other_nts = [i for i in range(4) if i != current_nt]
    new_nt = np.random.choice(other_nts)
    
    return mutate_position_to_base(siRNA_onehot, position, new_nt)


# =============================================================================
#   Expected-Effect Mutation Operator
# =============================================================================
def get_prediction(model, siRNA_enhanced, mRNA, siRNA_FM, mRNA_FM, td):
    """Get model prediction for a single input."""
    with torch.no_grad():
        out = model(siRNA_enhanced, mRNA, siRNA_FM, mRNA_FM, td)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        
        if logits.dim() == 2 and logits.size(-1) == 2:
            pred = torch.softmax(logits, dim=-1)[:, 1].item()
        elif logits.dim() == 1:
            pred = torch.sigmoid(logits).item()
        else:
            pred = logits.squeeze().item()
    return pred


def expected_effect_single_position(model, siRNA_onehot, position, 
                                     mRNA, siRNA_FM, mRNA_FM, td,
                                     orig_pred):
    """
    Compute expected effect of mutating a single position.
    
    Averages prediction change over ALL 3 possible substitutions.
    This reduces variance compared to single random substitution.
    
    Returns: mean |delta_pred| over 3 substitutions
    """
    if siRNA_onehot.dim() == 4:
        current_nt = siRNA_onehot[0, 0, position, :4].argmax().item()
    else:
        current_nt = siRNA_onehot[0, position, :4].argmax().item()
    
    delta_preds = []
    
    for new_nt in range(4):
        if new_nt == current_nt:
            continue
        
        mutated = mutate_position_to_base(siRNA_onehot, position, new_nt)
        mutated_enhanced = compute_physics_input_features(mutated)
        mutated_pred = get_prediction(model, mutated_enhanced, mRNA, siRNA_FM, mRNA_FM, td)
        
        delta_preds.append(abs(orig_pred - mutated_pred))
    
    return np.mean(delta_preds)


def expected_effect_position_set(model, siRNA_onehot, positions,
                                  mRNA, siRNA_FM, mRNA_FM, td,
                                  orig_pred):
    """
    Compute expected effect for a SET of positions (independent single-site).
    
    For each position, compute expected effect independently, then average.
    Avoids exponential blowup of 3^k multi-mutants.
    
    Returns: dict with mean, sum, and individual effects
    """
    effects = []
    for pos in positions:
        effect = expected_effect_single_position(
            model, siRNA_onehot, pos,
            mRNA, siRNA_FM, mRNA_FM, td,
            orig_pred
        )
        effects.append(effect)
    
    return {
        'mean': np.mean(effects),
        'sum': np.sum(effects),
        'individual': effects
    }


# =============================================================================
#   Nucleotide-Matched Random Baseline (v2.3 - NO poisoning)
# =============================================================================
def get_nucleotide_matched_random_positions(sequence, target_positions, n_samples=100):
    """
    Sample random position sets that MATCH the nucleotide composition of target_positions.
    
    v2.3: 
    - Try strict matching WITHOUT replacement (original behavior).
    - If not enough samples exist, fall back to sampling WITH replacement.
    - NEVER pad with target_positions (that would poison the baseline).
    - Returns whatever samples we could generate (may be < n_samples).
    
    Returns:
        List of position lists. Length may be < n_samples if matching is impossible.
    """
    target_bases = [sequence[p] for p in target_positions]
    k = len(target_positions)
    
    # Build index: base -> list of positions with that base
    base_to_positions = defaultdict(list)
    for i, base in enumerate(sequence):
        base_to_positions[base].append(i)
    
    # Count required bases
    base_counts = defaultdict(int)
    for b in target_bases:
        base_counts[b] += 1
    
    matched_samples = []
    seen_sets = set()  # Avoid duplicates
    
    # 1) Strict sampler (no replacement within sample)
    max_attempts = n_samples * 50
    attempts = 0
    while len(matched_samples) < n_samples and attempts < max_attempts:
        attempts += 1
        sampled = []
        success = True
        for base, count in base_counts.items():
            candidates = [p for p in base_to_positions.get(base, []) if p not in sampled]
            if len(candidates) < count:
                success = False
                break
            chosen = np.random.choice(candidates, size=count, replace=False).tolist()
            sampled.extend(chosen)
        if success and len(sampled) == k:
            sampled_tuple = tuple(sorted(sampled))
            # Exclude target set and duplicates
            if sampled_tuple != tuple(sorted(target_positions)) and sampled_tuple not in seen_sets:
                seen_sets.add(sampled_tuple)
                matched_samples.append(list(sampled_tuple))
    
    # 2) Fallback sampler (WITH replacement, for tight base distributions)
    if len(matched_samples) < n_samples:
        needed = n_samples - len(matched_samples)
        fallback_attempts = 0
        fallback_max_attempts = needed * 200
        
        while needed > 0 and fallback_attempts < fallback_max_attempts:
            fallback_attempts += 1
            sampled = []
            valid = True
            for base, count in base_counts.items():
                bucket = base_to_positions.get(base, [])
                if len(bucket) == 0:
                    valid = False
                    break
                replace = len(bucket) < count
                chosen = np.random.choice(bucket, size=count, replace=replace).tolist()
                sampled.extend(chosen)
            
            if not valid or len(sampled) != k:
                continue
            
            sampled_tuple = tuple(sorted(sampled))
            # Exclude target set and duplicates
            if sampled_tuple != tuple(sorted(target_positions)) and sampled_tuple not in seen_sets:
                seen_sets.add(sampled_tuple)
                matched_samples.append(list(sampled_tuple))
                needed -= 1
    
    # NEVER pad with target_positions - that would poison the baseline!
    # Just return what we have
    
    return matched_samples


# =============================================================================
#   Main Perturbation Test (v2.3)
# =============================================================================
def perturbation_test_v2(model, dataloader, device,
                         n_samples=100, k_positions=3, n_matched_samples=50):
    """
    Perturbation test with:
    1. Expected-effect mutation operator (averages over all 3 substitutions)
    2. Nucleotide-matched random baseline (controls for composition bias)
    
    Args:
        model: trained model
        dataloader: data loader
        device: torch device
        n_samples: number of samples to test
        k_positions: number of top positions to mutate
        n_matched_samples: number of matched random samples per sequence
    
    Returns:
        DataFrame with results for each sample
    """
    # Prepare model once (train mode for LSTM, dropout disabled)
    model = prepare_model_for_saliency(model)
    
    results = []
    sample_count = 0
    
    for batch in tqdm(dataloader, desc="Perturbation test v2"):
        if sample_count >= n_samples:
            break
        
        # Unpack batch
        siRNA_raw = batch[0]  # [B, 1, 19, 5]
        mRNA = batch[1].to(device)
        siRNA_FM = batch[2].to(device)
        mRNA_FM = batch[3].to(device)
        efficacy = batch[4].float().numpy()
        labels = batch[5].long().numpy()
        td = batch[6].to(device)
        
        batch_size = siRNA_raw.size(0)
        
        for i in range(batch_size):
            if sample_count >= n_samples:
                break
            
            # Single sample
            siRNA_single = siRNA_raw[i:i+1].to(device)
            mRNA_single = mRNA[i:i+1]
            siRNA_FM_single = siRNA_FM[i:i+1]
            mRNA_FM_single = mRNA_FM[i:i+1]
            td_single = td[i:i+1]
            
            # Extract sequence for nucleotide matching
            sequence = get_sequence_from_onehot(siRNA_single)
            
            # Add physics features
            siRNA_enhanced = compute_physics_input_features(siRNA_single)
            
            # Compute saliency
            try:
                saliency, orig_pred = compute_saliency(
                    model, siRNA_enhanced, mRNA_single, siRNA_FM_single, 
                    mRNA_FM_single, td_single, device
                )
            except Exception as e:
                print(f"Saliency failed for sample {sample_count}: {e}")
                continue
            
            # Get position importance
            pos_importance = get_position_importance(saliency)
            
            # Top-k and bottom-k positions
            sorted_positions = np.argsort(pos_importance)
            top_k_positions = sorted_positions[-k_positions:][::-1].tolist()
            bottom_k_positions = sorted_positions[:k_positions].tolist()
            
            # =================================================================
            # 1. EXPECTED-EFFECT for top-k positions
            # =================================================================
            topk_result = expected_effect_position_set(
                model, siRNA_single, top_k_positions,
                mRNA_single, siRNA_FM_single, mRNA_FM_single, td_single,
                orig_pred
            )
            topk_effect = topk_result['mean']
            
            # =================================================================
            # 2. EXPECTED-EFFECT for bottom-k positions
            # =================================================================
            bottomk_result = expected_effect_position_set(
                model, siRNA_single, bottom_k_positions,
                mRNA_single, siRNA_FM_single, mRNA_FM_single, td_single,
                orig_pred
            )
            bottomk_effect = bottomk_result['mean']
            
            # =================================================================
            # 3. NUCLEOTIDE-MATCHED random baseline (NO poisoning)
            # =================================================================
            matched_random_positions = get_nucleotide_matched_random_positions(
                sequence, top_k_positions, n_samples=n_matched_samples
            )
            
            # Compute effects only if we have samples (never poison with target)
            if len(matched_random_positions) == 0:
                matched_random_effect = np.nan
                matched_random_std = np.nan
                matched_n = 0
            else:
                matched_effects = []
                for rand_positions in matched_random_positions:
                    rand_result = expected_effect_position_set(
                        model, siRNA_single, rand_positions,
                        mRNA_single, siRNA_FM_single, mRNA_FM_single, td_single,
                        orig_pred
                    )
                    matched_effects.append(rand_result['mean'])
                
                matched_random_effect = np.mean(matched_effects)
                matched_random_std = np.std(matched_effects)
                matched_n = len(matched_effects)
            
            # =================================================================
            # 4. UNMATCHED random baseline (for comparison / bias quantification)
            # =================================================================
            unmatched_effects = []
            for _ in range(min(20, n_matched_samples)):
                rand_positions = np.random.choice(19, size=k_positions, replace=False).tolist()
                rand_result = expected_effect_position_set(
                    model, siRNA_single, rand_positions,
                    mRNA_single, siRNA_FM_single, mRNA_FM_single, td_single,
                    orig_pred
                )
                unmatched_effects.append(rand_result['mean'])
            
            unmatched_random_effect = np.mean(unmatched_effects)
            
            # Store results
            results.append({
                'sample_idx': sample_count,
                'efficacy': efficacy[i],
                'label': labels[i],
                'orig_pred': orig_pred,
                'sequence': ''.join(sequence),
                'topk_positions': top_k_positions,
                'topk_bases': [sequence[p] for p in top_k_positions],
                'topk_effect': topk_effect,
                'bottomk_effect': bottomk_effect,
                'matched_random_effect': matched_random_effect,
                'matched_random_std': matched_random_std,
                'matched_n': matched_n,  # Track actual number of matched samples
                'unmatched_random_effect': unmatched_random_effect,
                'topk_vs_matched': topk_effect - matched_random_effect if not np.isnan(matched_random_effect) else np.nan,
                'topk_vs_unmatched': topk_effect - unmatched_random_effect,
                'topk_vs_bottom': topk_effect - bottomk_effect,
                'composition_bias': unmatched_random_effect - matched_random_effect if not np.isnan(matched_random_effect) else np.nan,
                'pos_importance': pos_importance.tolist()
            })
            
            sample_count += 1
    
    return pd.DataFrame(results)


# =============================================================================
#   Statistical Analysis
# =============================================================================
def rank_biserial_signed_rank(a, b):
    """
    Proper rank-biserial correlation for Wilcoxon signed-rank test.
    Computed from W+ and W- (sum of positive and negative ranks).
    """
    diff = a - b
    diff = diff[diff != 0]  # ignore zeros like Wilcoxon
    if len(diff) == 0:
        return 0.0
    abs_diff = np.abs(diff)
    ranks = stats.rankdata(abs_diff)  # average ranks for ties
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom > 0 else 0.0


def cohens_d_paired(a, b):
    diff = a - b
    diff = diff[diff != 0]  # match Wilcoxon behavior
    if len(diff) < 2:
        return 0.0
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def compute_statistics(results_df):
    """Compute statistical tests and effect sizes."""
    
    topk = results_df['topk_effect'].values
    bottomk = results_df['bottomk_effect'].values
    matched = results_df['matched_random_effect'].values
    unmatched = results_df['unmatched_random_effect'].values
    matched_n = results_df['matched_n'].values
    
    # Remove NaN for matched comparisons
    valid_mask = ~np.isnan(matched)
    topk_valid = topk[valid_mask]
    matched_valid = matched[valid_mask]
    
    # Check minimum sample size for Wilcoxon
    if len(topk_valid) < 5:
        return {
            'n_samples': len(results_df),
            'n_valid': int(valid_mask.sum()),
            'error': 'Too few valid samples for statistical tests'
        }
    
    # Primary test: top-k vs matched random
    try:
        stat_matched, p_matched = stats.wilcoxon(topk_valid, matched_valid, alternative='greater')
    except ValueError as e:
        stat_matched, p_matched = np.nan, np.nan
    
    # Paired Cohen's d_z (appropriate for within-subject comparisons)
    d_matched = cohens_d_paired(topk_valid, matched_valid)
    
    # Rank-biserial correlation (proper calculation from signed ranks)
    rank_biserial = rank_biserial_signed_rank(topk_valid, matched_valid)
    
    # Secondary test: top-k vs bottom-k (use same valid_mask for consistency)
    topk_bottom_valid = topk[valid_mask]
    bottomk_valid = bottomk[valid_mask]
    try:
        stat_bottom, p_bottom = stats.wilcoxon(topk_bottom_valid, bottomk_valid, alternative='greater')
    except ValueError:
        stat_bottom, p_bottom = np.nan, np.nan
    d_bottom = cohens_d_paired(topk_bottom_valid, bottomk_valid)
    
    # Bias test: matched vs unmatched
    unmatched_valid = unmatched[valid_mask]
    try:
        stat_bias, p_bias = stats.wilcoxon(matched_valid, unmatched_valid)
    except ValueError:
        stat_bias, p_bias = np.nan, np.nan
    
    return {
        'n_samples': len(results_df),
        'n_valid': int(valid_mask.sum()),
        'matched_n_mean': float(np.mean(matched_n)),
        'matched_n_min': int(np.min(matched_n)) if len(matched_n) > 0 else 0,
        
        'topk_mean': float(np.mean(topk)),
        'topk_std': float(np.std(topk, ddof=1)),
        'bottomk_mean': float(np.mean(bottomk)),
        'bottomk_std': float(np.std(bottomk, ddof=1)),
        'matched_random_mean': float(np.nanmean(matched)),
        'matched_random_std': float(np.nanstd(matched)),
        'unmatched_random_mean': float(np.mean(unmatched)),
        'unmatched_random_std': float(np.std(unmatched, ddof=1)),
        
        'topk_vs_matched': {
            'statistic': float(stat_matched) if not np.isnan(stat_matched) else None,
            'p_value': float(p_matched) if not np.isnan(p_matched) else None,
            'effect_size_d': float(d_matched),
            'rank_biserial': float(rank_biserial),
            'frac_greater': float((topk_valid > matched_valid).mean())
        },
        'topk_vs_bottom': {
            'statistic': float(stat_bottom) if not np.isnan(stat_bottom) else None,
            'p_value': float(p_bottom) if not np.isnan(p_bottom) else None,
            'effect_size_d': float(d_bottom),
            'frac_greater': float((topk_bottom_valid > bottomk_valid).mean())
        },
        'composition_bias': {
            'p_value': float(p_bias) if not np.isnan(p_bias) else None,
            'mean_bias': float(np.mean(unmatched_valid) - np.mean(matched_valid))
        }
    }


# =============================================================================
#   Visualization
# =============================================================================
def plot_perturbation_results_v2(results_df, stats_dict, save_path, title_suffix=''):
    """Create publication-ready figure for v2 perturbation test."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Perturbation Test v2.3: Expected-Effect + Matched Baseline{title_suffix}', 
                 fontsize=14)
    
    # --- Panel A: Bar comparison (4 conditions) ---
    ax = axes[0, 0]
    
    means = [
        stats_dict['topk_mean'],
        stats_dict['matched_random_mean'],
        stats_dict['unmatched_random_mean'],
        stats_dict['bottomk_mean']
    ]
    stds = [
        stats_dict['topk_std'] / np.sqrt(stats_dict['n_samples']),
        stats_dict['matched_random_std'] / np.sqrt(stats_dict['n_valid']),
        stats_dict['unmatched_random_std'] / np.sqrt(stats_dict['n_samples']),
        stats_dict['bottomk_std'] / np.sqrt(stats_dict['n_samples'])
    ]
    
    labels = ['Top-K', 'Matched\nRandom', 'Unmatched\nRandom', 'Bottom-K']
    colors = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c']
    
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel('Expected Effect (mean |Δpred|)', fontsize=11)
    ax.set_title('A) Expected Effect by Mutation Type', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add significance annotation
    tv_m = stats_dict['topk_vs_matched']
    p_val = tv_m['p_value']
    if p_val is not None:
        sig_text = f'p = {p_val:.2e}' if p_val < 0.001 else f'p = {p_val:.3f}'
        stars = '*' * (int(p_val < 0.05) + int(p_val < 0.01) + int(p_val < 0.001))
        ax.annotate(f'{sig_text} {stars}', xy=(0.5, max(means) * 1.15), fontsize=10, ha='center')
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Panel B: Top-k vs Matched Random scatter ---
    ax = axes[0, 1]
    
    valid_mask = ~results_df['matched_random_effect'].isna()
    if valid_mask.sum() > 0:
        scatter = ax.scatter(
            results_df.loc[valid_mask, 'matched_random_effect'], 
            results_df.loc[valid_mask, 'topk_effect'],
            alpha=0.5, 
            c=results_df.loc[valid_mask, 'efficacy'].astype(float), 
            cmap='RdYlGn', s=30
        )
        
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal effect')
        
        frac_above = tv_m['frac_greater']
        ax.annotate(f'{frac_above:.1%} above diagonal\nd_z = {tv_m["effect_size_d"]:.2f}\nr = {tv_m["rank_biserial"]:.2f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, va='top')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('True Efficacy')
    
    ax.set_xlabel('Matched Random Effect', fontsize=11)
    ax.set_ylabel('Top-K Effect', fontsize=11)
    ax.set_title('B) Top-K vs Matched Random (per sample)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # --- Panel C: Composition bias visualization ---
    ax = axes[1, 0]
    
    bias_values = results_df['composition_bias'].dropna().values
    
    if len(bias_values) > 0:
        ax.hist(bias_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No bias')
        ax.axvline(x=np.mean(bias_values), color='orange', linestyle='-', linewidth=2, 
                   label=f'Mean bias = {np.mean(bias_values):.4f}')
        ax.legend()
    
    ax.set_xlabel('Composition Bias (Unmatched - Matched)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('C) Nucleotide Composition Bias', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Panel D: Position frequency in top-K ---
    ax = axes[1, 1]
    
    position_counts = np.zeros(19)
    for positions in results_df['topk_positions']:
        for pos in positions:
            position_counts[pos] += 1
    position_counts = position_counts / len(results_df)
    
    # FIXED: Seed region = positions 2-8 (1-indexed) = indices 1-7 (0-indexed)
    colors_pos = ['#d62728' if 1 <= i <= 7 else '#1f77b4' for i in range(19)]
    ax.bar(range(1, 20), position_counts, color=colors_pos, alpha=0.8)
    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel('Frequency in Top-K', fontsize=11)
    ax.set_title('D) Most Important Positions (red = seed region)', fontsize=12)
    ax.set_xticks(range(1, 20))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {save_path}")


def print_results(stats_dict):
    """Pretty-print validation results."""
    
    print("\n" + "="*70)
    print("PERTURBATION TEST v2.3 RESULTS")
    print("  (Expected-Effect Operator + Nucleotide-Matched Baseline)")
    print("="*70)
    
    if 'error' in stats_dict:
        print(f"\nERROR: {stats_dict['error']}")
        return
    
    print(f"\nSamples tested: {stats_dict['n_samples']} ({stats_dict['n_valid']} with valid matched baseline)")
    print(f"Matched samples per sequence: mean={stats_dict['matched_n_mean']:.1f}, min={stats_dict['matched_n_min']}")
    
    print("\n--- Expected Effects (mean +/- std) ---")
    print(f"  Top-K positions:      {stats_dict['topk_mean']:.4f} +/- {stats_dict['topk_std']:.4f}")
    print(f"  Matched random:       {stats_dict['matched_random_mean']:.4f} +/- {stats_dict['matched_random_std']:.4f}")
    print(f"  Unmatched random:     {stats_dict['unmatched_random_mean']:.4f} +/- {stats_dict['unmatched_random_std']:.4f}")
    print(f"  Bottom-K positions:   {stats_dict['bottomk_mean']:.4f} +/- {stats_dict['bottomk_std']:.4f}")
    
    print("\n--- Primary Test: Top-K vs Matched Random ---")
    tv_m = stats_dict['topk_vs_matched']
    if tv_m['statistic'] is not None:
        print(f"  Wilcoxon W = {tv_m['statistic']:.1f}")
        print(f"  p-value = {tv_m['p_value']:.2e} (one-sided)")
    print(f"  Cohen's d_z = {tv_m['effect_size_d']:.3f} (paired)")
    print(f"  Rank-biserial r = {tv_m['rank_biserial']:.3f}")
    print(f"  Fraction top-K > matched: {tv_m['frac_greater']:.1%}")
    
    if tv_m['p_value'] is not None and tv_m['p_value'] < 0.05 and tv_m['effect_size_d'] > 0.2:
        print("  PASS: Saliency identifies causally relevant positions")
    else:
        print("  FAIL: Saliency not significantly better than matched random")
    
    print("\n--- Secondary Test: Top-K vs Bottom-K ---")
    tv_b = stats_dict['topk_vs_bottom']
    if tv_b['statistic'] is not None:
        print(f"  Wilcoxon W = {tv_b['statistic']:.1f}, p = {tv_b['p_value']:.2e}")
    print(f"  Cohen's d_z = {tv_b['effect_size_d']:.3f} (paired)")
    
    print("\n--- Composition Bias (Unmatched - Matched) ---")
    cb = stats_dict['composition_bias']
    print(f"  Mean bias: {cb['mean_bias']:.4f}")
    if cb['p_value'] is not None:
        print(f"  p-value: {cb['p_value']:.2e}")
    if abs(cb['mean_bias']) > 0.005:
        print("  Note: Nucleotide matching corrects measurable composition bias")
    
    print("\n" + "="*70)


# =============================================================================
#   Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Perturbation test v2.3 with expected-effect and matched baseline')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='Hu', help='Dataset name')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to data directory')
    parser.add_argument('--fold', type=int, default=None, help='Fold number (1-5) to use as test set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (MUST match training Args.seed)')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of samples to test')
    parser.add_argument('--k_positions', type=int, default=3, help='Number of positions to mutate')
    parser.add_argument('--n_matched', type=int, default=50, help='Matched random samples per sequence')
    parser.add_argument('--output_dir', type=str, default='perturbation_results_v2', help='Output directory')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')
    
    args = parser.parse_args()
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running perturbation test v2.3")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Fold: {args.fold}")
    print(f"  Seed: {args.seed}")
    print(f"  N samples: {args.n_samples}")
    print(f"  K positions: {args.k_positions}")
    print(f"  N matched samples: {args.n_matched}")
    print(f"  Device: {device}")
    
    # Import model and data loader
    from scripts.model import Oligo
    from scripts.loader import data_process_loader
    
    # Load data - MUST match training preprocessing exactly
    dataset_df = pd.read_csv(f'{args.data_path}{args.dataset}.csv', dtype=str)
    
    if args.fold is not None:
        # Match training EXACTLY:
        # 1. Shuffle with training seed (Args.seed)
        # 2. Reset index
        # 3. Apply KFold with random_state=42
        dataset_df = shuffle(dataset_df, random_state=args.seed).reset_index(drop=True)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(dataset_df))
        _, test_indices = splits[args.fold - 1]  # fold is 1-indexed
        
        test_df = dataset_df.iloc[test_indices].reset_index(drop=True)
        print(f"  Fold {args.fold} test set: {len(test_df)} samples (fold-matched)")
    else:
        # Fallback to random sample (for quick testing, NOT fold-matched)
        dataset_df = shuffle(dataset_df, random_state=args.seed).reset_index(drop=True)
        test_df = dataset_df.head(min(args.n_samples * 2, len(dataset_df))).reset_index(drop=True)
        print(f"  Random sample: {len(test_df)} samples (NOT fold-matched)")
    
    # Create dataloader - indices are 0..len-1 after reset_index(drop=True)
    dataloader = DataLoader(
        data_process_loader(
            test_df.index.values,      # [0, 1, 2, ...] after reset_index
            test_df.label.values,
            test_df.y.values,
            test_df,
            args.dataset,
            args.data_path
        ),
        batch_size=1, shuffle=False, num_workers=0
    )
    
    # Load model - MUST match training architecture
    model = Oligo(
        vocab_size=26,
        embedding_dim=128,
        lstm_dim=32,
        n_head=8,
        n_layers=1,
        lm1=19,
        lm2=19
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    print(f"\nModel loaded. Running perturbation test v2.3...")
    
    # Run test
    results_df = perturbation_test_v2(
        model, dataloader, device,
        n_samples=args.n_samples, 
        k_positions=args.k_positions,
        n_matched_samples=args.n_matched
    )
    
    # Compute statistics
    stats_dict = compute_statistics(results_df)
    
    # Save results
    results_df.to_csv(f'{args.output_dir}/perturbation_results_v2.csv', index=False)
    
    import json
    with open(f'{args.output_dir}/statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    # Plot
    plot_perturbation_results_v2(
        results_df, stats_dict,
        f'{args.output_dir}/perturbation_validation_v2.png',
        f' ({args.dataset}, fold={args.fold}, k={args.k_positions})'
    )
    
    # Print results
    print_results(stats_dict)
    
    print(f"\nResults saved to: {args.output_dir}/")
    
    return results_df, stats_dict


if __name__ == '__main__':
    main()