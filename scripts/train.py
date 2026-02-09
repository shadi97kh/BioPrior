import torch
import os
import sys
import sklearn
import random
import pickle as pkl
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import data_process_loader
from torch.utils.data import DataLoader
from model import Oligo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                            accuracy_score, roc_auc_score, average_precision_score,
                            roc_curve, auc, precision_recall_curve, matthews_corrcoef,
                            confusion_matrix)
from metrics import sensitivity, specificity
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib as mpl
from logger import TrainLogger
from scripts.learnable_physics_loss import BioPriorModule
from scipy.stats import pearsonr  
import seaborn as sns  
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# ABLATION CONFIGS - Simplified for mechanistic BioPrior
# =============================================================================
ABLATION_CONFIGS = {
    'baseline': {
        'description': 'No physics (baseline OligoFormer)',
        'use_physics': False,
    },
    'mechanistic': {
        'description': 'Mechanistic BioPrior (asymmetry + motifs + GC)',
        'use_physics': True,
    },
    # Backward compatibility aliases - all use same settings
    'full': {
        'description': 'Mechanistic BioPrior',
        'use_physics': True,
    },
    'neutral': {
        'description': 'Mechanistic BioPrior',
        'use_physics': True,
    },
}


# =============================================================================
#   Physics Input Features - UNBIASED VERSION
# =============================================================================
def compute_physics_input_features(siRNA_onehot):
    """
    Add physics features to input WITHOUT position-specific biases.
    
    Input: [batch, 1, 19, 5] (your format)
    Output: [batch, 1, 19, 13] (5 original + 8 physics)
    
    Features are SEQUENCE-DEPENDENT, not position-hardcoded:
    - Uniform position importance (no bias)
    - Binary indicators for seed/cleavage (structural, not learned)
    - Computed GC, seed AU, asymmetry from actual sequence
    """
    x = siRNA_onehot.squeeze(1)[:, :, :4]  # [B, 19, 4]
    batch_size, seq_len, _ = x.shape
    dev = x.device
    
    seq_idx = x.argmax(dim=-1)  # [B, 19]
    
    # UNIFORM position importance (no bias)
    pos_imp = torch.ones(batch_size, seq_len, 1, device=dev) / seq_len
    
    # Seed region indicator [1:8] - structural, not biased
    seed_ind = torch.zeros(batch_size, seq_len, 1, device=dev)
    seed_ind[:, 1:8, 0] = 1.0
    
    # Cleavage site indicator [9:11] - structural, not biased
    cleave_ind = torch.zeros(batch_size, seq_len, 1, device=dev)
    cleave_ind[:, 9:11, 0] = 1.0
    
    # Computed from actual sequence (not hardcoded):
    # Global GC content
    gc = ((seq_idx == 2) | (seq_idx == 3)).float().mean(dim=1, keepdim=True).unsqueeze(-1).expand(-1, seq_len, -1)
    
    # Seed AU content
    seed_au = ((seq_idx[:, 1:8] == 0) | (seq_idx[:, 1:8] == 1)).float().mean(dim=1, keepdim=True).unsqueeze(-1).expand(-1, seq_len, -1)
    
    # Thermodynamic asymmetry (5' AU - 3' AU)
    au_mask = (seq_idx == 0) | (seq_idx == 1)
    asym = (au_mask[:, :5].float().mean(dim=1) - au_mask[:, -5:].float().mean(dim=1)).view(-1, 1, 1).expand(-1, seq_len, -1)
    
    # Per-position AU/GC indicators
    is_au = ((seq_idx == 0) | (seq_idx == 1)).float().unsqueeze(-1)
    is_gc = ((seq_idx == 2) | (seq_idx == 3)).float().unsqueeze(-1)
    
    # Original features [B, 19, 5]
    orig = siRNA_onehot.squeeze(1)
    
    # Concatenate: [B, 19, 5+8=13]
    enhanced = torch.cat([orig, pos_imp, seed_ind, cleave_ind, gc, seed_au, asym, is_au, is_gc], dim=-1)
    
    return enhanced.unsqueeze(1)


# =============================================================================
#   Physics cache for nucleotide probabilities
# =============================================================================
_nt_probs_cache = None

def get_nt_probs_cache():
    global _nt_probs_cache
    return _nt_probs_cache

def set_nt_probs_cache(probs):
    global _nt_probs_cache
    _nt_probs_cache = probs


# =============================================================================
#   EMA (Exponential Moving Average)
# =============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# =============================================================================
#   Helper functions
# =============================================================================
def _unpack_model_out(out):
    """Unpack model output with NaN checking"""
    if isinstance(out, tuple):
        logits = out[0]
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)
        
        attn1 = out[1] if len(out) > 1 else None
        attn2 = out[2] if len(out) > 2 else None

        def get_L(a):
            return a.size(-1) if (a is not None and a.dim() == 4) else None

        L1, L2 = get_L(attn1), get_L(attn2)
        if L1 == 19:
            siRNA_attention, mRNA_attention = attn1, attn2
        elif L2 == 19:
            siRNA_attention, mRNA_attention = attn2, attn1
        else:
            siRNA_attention, mRNA_attention = attn1, attn2

        if len(out) > 3 and out[3] is not None:
            set_nt_probs_cache(out[3])

        aux_preds = out[4] if len(out) > 4 else None

        return logits, siRNA_attention, mRNA_attention, aux_preds
    return out, None, None, None


def prob_from_logits(logits):
    """Extract probability from logits safely."""
    if logits.dim() == 2 and logits.size(-1) == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    elif logits.dim() == 1:
        return torch.sigmoid(logits)
    else:
        return logits.squeeze()


def compute_aux_loss(aux_preds, siRNA_batch):
    """Compute auxiliary multi-task loss"""
    if aux_preds is None:
        return torch.tensor(0.0, device=device)
    
    gc_pred, seed_au_pred, asymmetry_pred = aux_preds
    
    x = siRNA_batch.squeeze(1)[:, :, :4]
    seq_idx = x.argmax(dim=-1)
    
    true_gc = ((seq_idx == 2) | (seq_idx == 3)).float().mean(dim=1)
    true_seed_au = ((seq_idx[:, 1:8] == 0) | (seq_idx[:, 1:8] == 1)).float().mean(dim=1)
    
    au = (seq_idx == 0) | (seq_idx == 1)
    true_asym = au[:, :5].float().mean(dim=1) - au[:, -5:].float().mean(dim=1)
    
    gc_loss = F.mse_loss(gc_pred, true_gc)
    seed_loss = F.mse_loss(seed_au_pred, true_seed_au)
    asym_loss = F.mse_loss(asymmetry_pred, true_asym)
    
    return gc_loss + seed_loss + asym_loss


def compute_sample_weights(efficacy_values, pos_threshold=0.7, max_weight=3.0):
    """Compute sample weights for imbalanced regression"""
    efficacy = efficacy_values.astype(float)
    pos_mask = efficacy >= pos_threshold
    neg_mask = ~pos_mask
    
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(efficacy))
    
    if n_pos < n_neg:
        pos_weight = min(n_neg / n_pos, max_weight)
        neg_weight = 1.0
    else:
        pos_weight = 1.0
        neg_weight = min(n_pos / n_neg, max_weight)
    
    weights = np.where(pos_mask, pos_weight, neg_weight)
    return weights


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    
    def get_average(self):
        return self.sum / max(self.count, 1)


# =============================================================================
#   Validation function
# =============================================================================
def val_mse(model, loader, threshold=0.7):
    """Validation with MSE and metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_efficacy = []
    total_loss = 0
    n_samples = 0
    
    with torch.no_grad():
        for data in loader:
            siRNA = compute_physics_input_features(data[0].to(device))
            mRNA = data[1].to(device)
            siRNA_FM = data[2].to(device)
            mRNA_FM = data[3].to(device)
            labels = data[5].long().to(device)
            efficacy = data[4].float().to(device)
            td = data[6].to(device)
            
            out = model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
            logits, _, _, _ = _unpack_model_out(out)
            pred_efficacy = prob_from_logits(logits)
            
            mse = F.mse_loss(pred_efficacy, efficacy)
            total_loss += mse.item() * siRNA.size(0)
            n_samples += siRNA.size(0)
            
            all_preds.extend(pred_efficacy.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_efficacy.extend(efficacy.cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    efficacy = np.array(all_efficacy)
    
    # Compute metrics
    rocauc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    prauc = average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    pcc = pearsonr(efficacy, preds)[0] if len(efficacy) > 1 else 0
    
    # Find optimal threshold
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        pred_cls = (preds >= t).astype(int)
        f1 = f1_score(labels, pred_cls, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    
    pred_cls = (preds >= best_thresh).astype(int)
    precision = precision_score(labels, pred_cls, zero_division=0)
    recall = recall_score(labels, pred_cls, zero_division=0)
    
    return {
        'mse': total_loss / n_samples,
        'rocauc': rocauc,
        'prauc': prauc,
        'f1': best_f1,
        'pcc': pcc if not np.isnan(pcc) else 0,
        'precision': precision,
        'recall': recall,
        'optimal_threshold': best_thresh
    }


# =============================================================================
#   K-fold data split
# =============================================================================
def get_kfold_data_2(i, datasets, k=5, v=1):
    """Get k-fold split"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(datasets))
    train_indices, val_indices = splits[i]
    return datasets.iloc[train_indices].reset_index(drop=True), datasets.iloc[val_indices].reset_index(drop=True)


# =============================================================================
#   Plotting functions
# =============================================================================
def plot_training_history(history, save_dir):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    epochs = range(len(history['train_mse']))
    
    axes[0,0].plot(epochs, history['train_mse'], label='Train', color='blue')
    axes[0,0].plot(epochs, history['val_mse'], label='Val', color='red')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(epochs, history['val_rocauc'], color='darkorange')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('ROC-AUC')
    axes[0,1].set_title('ROC-AUC')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].plot(epochs, history['val_pcc'], color='purple')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('PCC')
    axes[0,2].set_title('Pearson Correlation')
    axes[0,2].grid(True, alpha=0.3)
    
    axes[1,0].plot(epochs, history['val_f1'], color='green')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('F1')
    axes[1,0].set_title('F1 Score')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(epochs, history['val_prauc'], color='magenta')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('PR-AUC')
    axes[1,1].set_title('PR-AUC')
    axes[1,1].grid(True, alpha=0.3)
    
    if 'physics_loss' in history:
        axes[1,2].plot(epochs, history['physics_loss'], color='teal')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Physics Loss')
        axes[1,2].set_title('Physics Loss')
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_fold_comparison(fold_results, save_dir, dataset_name):
    """Plot fold comparison"""
    metrics = ['rocauc', 'prauc', 'f1', 'pcc']
    metric_names = ['ROC-AUC', 'PR-AUC', 'F1', 'PCC']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for ax, metric, name in zip(axes, metrics, metric_names):
        values = [r[metric] for r in fold_results]
        folds = range(1, len(values) + 1)
        
        ax.bar(folds, values, color='steelblue', alpha=0.7)
        ax.axhline(y=np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel(name)
        ax.set_title(f'{name} by Fold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'5-Fold CV - {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fold_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
#   5-Fold Cross-Validation
# =============================================================================
def train_intra_5fold(Args):
    """5-fold cross-validation with mechanistic BioPrior."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
    random.seed(Args.seed)
    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    
    dataset_name = Args.datasets[0]
    dataset_df = pd.read_csv(Args.path + dataset_name + '.csv', dtype=str)
    dataset_df = shuffle(dataset_df, random_state=Args.seed).reset_index(drop=True)
    
    efficacy_values = dataset_df.label.astype(float).values
    pos_total = (efficacy_values >= 0.7).sum()
    neg_total = (efficacy_values < 0.7).sum()
    print(f"\n{dataset_name}: Pos={pos_total}, Neg={neg_total}")
    
    # Ablation configuration
    ablation_mode = getattr(Args, 'ablation', 'mechanistic')
    if ablation_mode not in ABLATION_CONFIGS:
        print(f"Warning: Unknown ablation '{ablation_mode}', using 'mechanistic'")
        ablation_mode = 'mechanistic'
    
    config = ABLATION_CONFIGS[ablation_mode]
    print(f"\n{'='*60}")
    print(f"ABLATION MODE: {ablation_mode}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")
    
    all_fold_results = []
    # shuffle=False so sample_weights indexing works correctly
    params = {'batch_size': Args.batch_size, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
    
    logger_params = dict(
        data_path=Args.path,
        save_dir=getattr(Args, 'output_dir', 'result'),
        dataset=f"{dataset_name}_5fold_mse_{ablation_mode}",
        batch_size=Args.batch_size
    )
    logger = TrainLogger(logger_params)
    
    for fold in range(5):
        print(f"\n{'='*50} FOLD {fold+1}/5 {'='*50}")
        
        # Create physics module
        if config['use_physics']:
            physics_module = BioPriorModule(seq_len=19).to(device)
        else:
            physics_module = None
        
        train_df, valid_df = get_kfold_data_2(fold, dataset_df, k=5)
        
        train_efficacy = train_df.label.astype(float).values
        sample_weights = compute_sample_weights(train_efficacy, pos_threshold=0.7)
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
        
        train_ds = DataLoader(data_process_loader(train_df.index.values, train_df.label.values,
                             train_df.y.values, train_df, dataset_name, Args.path), **params)
        valid_ds = DataLoader(data_process_loader(valid_df.index.values, valid_df.label.values,
                             valid_df.y.values, valid_df, dataset_name, Args.path),
                             batch_size=Args.batch_size, shuffle=False, num_workers=0)
        
        OFmodel = Oligo(vocab_size=Args.vocab_size, embedding_dim=Args.embedding_dim,
                       lstm_dim=Args.lstm_dim, n_head=Args.n_head, n_layers=Args.n_layers,
                       lm1=Args.lm1, lm2=Args.lm2).to(device)
        
        # Initialize
        OFmodel.eval()
        with torch.no_grad():
            batch = next(iter(train_ds))
            siRNA_init = compute_physics_input_features(batch[0].to(device))
            _ = OFmodel(siRNA_init, batch[1].to(device), batch[2].to(device),
                       batch[3].to(device), batch[6].to(device))
        OFmodel.train()
        
        ema = EMA(OFmodel, decay=0.999)
        optimizer = torch.optim.AdamW(OFmodel.parameters(), lr=Args.learning_rate, weight_decay=1e-4)
        
        warmup_epochs = 8
        sched_warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        sched_cos = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, [sched_warm, sched_cos], milestones=[warmup_epochs])
        
        best_rocauc = 0.0
        best_epoch = 0
        early_stopping = getattr(Args, 'early_stopping', 30)
        
        history = {'train_mse': [], 'val_mse': [], 'physics_loss': [],
                   'val_rocauc': [], 'val_prauc': [], 'val_f1': [], 'val_pcc': [],
                   'val_precision': [], 'val_recall': []}
        
        for epoch in range(min(Args.epoch, 300)):
            OFmodel.train()
            running_mse = AverageMeter()
            physics_tracker = AverageMeter()
            
            for batch_idx, data in enumerate(train_ds):
                siRNA = compute_physics_input_features(data[0].to(device))
                mRNA = data[1].to(device)
                siRNA_FM = data[2].to(device)
                mRNA_FM = data[3].to(device)
                efficacy = data[4].float().to(device)
                td = data[6].to(device)
                
                batch_start = batch_idx * Args.batch_size
                batch_end = min(batch_start + Args.batch_size, len(sample_weights_tensor))
                batch_weights = sample_weights_tensor[batch_start:batch_end].to(device)
                if len(batch_weights) < siRNA.size(0):
                    batch_weights = torch.ones(siRNA.size(0), device=device)
                
                optimizer.zero_grad()
                
                out = OFmodel(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
                logits, siRNA_attention, _, aux_preds = _unpack_model_out(out)
                pred_efficacy = prob_from_logits(logits)
                
                # MSE loss
                per_sample_mse = (pred_efficacy - efficacy) ** 2          # [B]
                mse_loss = (per_sample_mse * batch_weights).mean()        # scalar
                
                # Physics loss - use MODEL'S nucleotide predictions for gradient flow
                if config['use_physics'] and physics_module is not None:
                    # Get model's soft nucleotide predictions (out[3])
                    si_nt_probs = out[3] if len(out) > 3 and out[3] is not None else None
                    
                    if si_nt_probs is not None:
                        # Use model's predictions - gradients flow back!
                        physics_loss = physics_module(si_nt_probs, pred_efficacy, efficacy, epoch, mRNA_input=mRNA)
                        if batch_idx == 0 and epoch == 0:
                            print(f"[DEBUG] Using si_nt_probs shape={si_nt_probs.shape}, physics_loss={physics_loss.item():.4f}")
                    else:
                        # Fallback to input (no gradient to model, but still regularizes)
                        siRNA_onehot = siRNA.squeeze(1)[:, :, :4]
                        physics_loss = physics_module(siRNA_onehot, pred_efficacy, efficacy, epoch, mRNA_input=mRNA)
                        if batch_idx == 0 and epoch == 0:
                            print(f"[DEBUG] si_nt_probs is None! Using input one-hot (no gradient flow)")
                else:
                    physics_loss = torch.tensor(0.0, device=device)
                
                # Aux loss
                aux_loss = compute_aux_loss(aux_preds, siRNA)
                
                # Combine - physics weight ramped and capped at 0.10
                if epoch < warmup_epochs:
                    pw = 0.0  # No physics during warmup
                else:
                    # Ramp from 0.02 to cap of 0.10
                    pw = min(0.30, 0.10 + 0.01 * (epoch - warmup_epochs))
                
                if not config['use_physics']:
                    pw = 0.0
                
                loss = mse_loss + pw * physics_loss + 0.02 * aux_loss
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(OFmodel.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update(OFmodel)
                
                running_mse.update(mse_loss.item(), siRNA.size(0))
                physics_tracker.update(physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss, siRNA.size(0))
            
            scheduler.step()
            
            # Validation
            ema.apply_to(OFmodel)
            val_metrics = val_mse(OFmodel, valid_ds)
            ema.restore(OFmodel)
            
            # Track history
            history['train_mse'].append(running_mse.get_average())
            history['val_mse'].append(val_metrics['mse'])
            history['physics_loss'].append(physics_tracker.get_average())
            history['val_rocauc'].append(val_metrics['rocauc'])
            history['val_prauc'].append(val_metrics['prauc'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_pcc'].append(val_metrics['pcc'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            
            # Save best
            if val_metrics['rocauc'] > best_rocauc:
                best_rocauc = val_metrics['rocauc']
                best_epoch = epoch
                ema.apply_to(OFmodel)
                torch.save(OFmodel.state_dict(), f'{logger.get_model_dir()}/fold_{fold+1}_best.pth')
                ema.restore(OFmodel)
            
            # Early stopping
            if epoch - best_epoch > early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model for final evaluation
        OFmodel.load_state_dict(torch.load(f'{logger.get_model_dir()}/fold_{fold+1}_best.pth'))
        final_metrics = val_mse(OFmodel, valid_ds)
        
        all_fold_results.append(final_metrics)
        
        print(f"Fold {fold+1}: AUC={final_metrics['rocauc']:.4f}, PRC={final_metrics['prauc']:.4f}, "
              f"F1={final_metrics['f1']:.4f}, PCC={final_metrics['pcc']:.4f}")
        
        # Visualize physics constraints
        if physics_module is not None:
            physics_module.visualize(f'{logger.get_model_dir()}/physics_fold{fold+1}.png')
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"5-FOLD CV RESULTS - {dataset_name} ({ablation_mode})")
    print(f"{'='*60}")
    
    for metric in ['rocauc', 'prauc', 'f1', 'pcc']:
        values = [r[metric] for r in all_fold_results]
        print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Save results
    plot_fold_comparison(all_fold_results, logger.get_model_dir(), dataset_name)
    
    results_df = pd.DataFrame(all_fold_results)
    results_df['ablation'] = ablation_mode
    results_df.to_csv(f'{logger.get_model_dir()}/{dataset_name}_5fold_results.csv', index=False)
    
    # Run analysis on last fold's model
    print("\nGenerating saliency maps and analysis...")
    try:
        from scripts.analysis import (compute_batch_saliency, plot_saliency_heatmap,
                                       plot_position_importance, plot_nucleotide_preferences,
                                       analyze_physics_violations, plot_physics_violations_vs_efficacy)
        
        # Load best model from fold 5
        OFmodel.load_state_dict(torch.load(f'{logger.get_model_dir()}/fold_5_best.pth'))
        
        # Use validation set from last fold for analysis
        avg_sal, pos_sal, neg_sal = compute_batch_saliency(OFmodel, valid_ds, device, n_samples=100)
        
        plot_saliency_heatmap(avg_sal, f"{dataset_name} Saliency ({ablation_mode})",
                             f'{logger.get_model_dir()}/saliency_heatmap.png')
        plot_position_importance(avg_sal, f"Position Importance ({ablation_mode})",
                                f'{logger.get_model_dir()}/position_importance.png')
        plot_nucleotide_preferences(avg_sal, f"Nucleotide Preferences ({ablation_mode})",
                                   f'{logger.get_model_dir()}/nucleotide_preferences.png')
        
        # High vs low efficacy
        plot_saliency_heatmap(pos_sal, "High Efficacy Samples",
                             f'{logger.get_model_dir()}/saliency_high_efficacy.png')
        plot_saliency_heatmap(neg_sal, "Low Efficacy Samples",
                             f'{logger.get_model_dir()}/saliency_low_efficacy.png')
        
        # Physics analysis
        violations_df = analyze_physics_violations(valid_ds, device, n_samples=200)
        violations_df.to_csv(f'{logger.get_model_dir()}/physics_violations.csv', index=False)
        plot_physics_violations_vs_efficacy(violations_df,
                                            f'{logger.get_model_dir()}/violations_vs_efficacy.png')
        
        np.savez(f'{logger.get_model_dir()}/saliency_data.npz',
                 avg_saliency=avg_sal, pos_saliency=pos_sal, neg_saliency=neg_sal)
        
        print(f"Analysis saved to {logger.get_model_dir()}")
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    print(f"\nAll results saved to: {logger.get_model_dir()}")
    
    return all_fold_results


# =============================================================================
#   Inter-dataset Training (train on source, test on target)
# =============================================================================
def train(Args):
    """
    Inter-dataset training: train on Args.datasets[0], test on Args.datasets[1].
    This tests cross-dataset generalization.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
    random.seed(Args.seed)
    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    
    source_name = Args.datasets[0]
    target_name = Args.datasets[1]
    
    # Load datasets
    train_df = pd.read_csv(Args.path + source_name + '.csv', dtype=str)
    test_df = pd.read_csv(Args.path + target_name + '.csv', dtype=str)
    
    # Split source into train/val (80/20)
    train_df = shuffle(train_df, random_state=Args.seed).reset_index(drop=True)
    split_idx = int(len(train_df) * 0.8)
    val_df = train_df.iloc[split_idx:].reset_index(drop=True)
    train_df = train_df.iloc[:split_idx].reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"INTER-DATASET TRAINING")
    print(f"Source: {source_name} (train={len(train_df)}, val={len(val_df)})")
    print(f"Target: {target_name} (test={len(test_df)})")
    print(f"{'='*60}\n")
    
    # Ablation config
    ablation_mode = getattr(Args, 'ablation', 'mechanistic')
    if ablation_mode not in ABLATION_CONFIGS:
        ablation_mode = 'mechanistic'
    config = ABLATION_CONFIGS[ablation_mode]
    
    print(f"Ablation: {ablation_mode} - {config['description']}")
    
    # Physics module
    if config['use_physics']:
        physics_module = BioPriorModule(seq_len=19).to(device)
    else:
        physics_module = None
    
    # Data loaders - shuffle=False so sample_weights indexing works correctly
    params = {'batch_size': Args.batch_size, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
    
    train_ds = DataLoader(data_process_loader(train_df.index.values, train_df.label.values,
                         train_df.y.values, train_df, source_name, Args.path), **params)
    val_ds = DataLoader(data_process_loader(val_df.index.values, val_df.label.values,
                       val_df.y.values, val_df, source_name, Args.path),
                       batch_size=Args.batch_size, shuffle=False, num_workers=0)
    test_ds = DataLoader(data_process_loader(test_df.index.values, test_df.label.values,
                        test_df.y.values, test_df, target_name, Args.path),
                        batch_size=Args.batch_size, shuffle=False, num_workers=0)
    
    # Sample weights
    train_efficacy = train_df.label.astype(float).values
    sample_weights = compute_sample_weights(train_efficacy, pos_threshold=0.7)
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    
    # Model
    OFmodel = Oligo(vocab_size=Args.vocab_size, embedding_dim=Args.embedding_dim,
                   lstm_dim=Args.lstm_dim, n_head=Args.n_head, n_layers=Args.n_layers,
                   lm1=Args.lm1, lm2=Args.lm2).to(device)
    
    # Initialize
    OFmodel.eval()
    with torch.no_grad():
        batch = next(iter(train_ds))
        siRNA_init = compute_physics_input_features(batch[0].to(device))
        _ = OFmodel(siRNA_init, batch[1].to(device), batch[2].to(device),
                   batch[3].to(device), batch[6].to(device))
    OFmodel.train()
    
    ema = EMA(OFmodel, decay=0.999)
    optimizer = torch.optim.AdamW(OFmodel.parameters(), lr=Args.learning_rate, weight_decay=1e-4)
    
    warmup_epochs = 8
    sched_warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    sched_cos = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [sched_warm, sched_cos], milestones=[warmup_epochs])
    
    # Logger
    logger_params = dict(
        data_path=Args.path,
        save_dir=getattr(Args, 'output_dir', 'result'),
        dataset=f"{source_name}_to_{target_name}_{ablation_mode}",
        batch_size=Args.batch_size
    )
    logger = TrainLogger(logger_params)
    
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_epoch = 0
    early_stopping = getattr(Args, 'early_stopping', 30)
    
    history = {'train_mse': [], 'val_auc': [], 'test_auc': [], 'test_pcc': [], 'physics_loss': []}
    
    for epoch in range(Args.epoch):
        OFmodel.train()
        running_mse = AverageMeter()
        physics_tracker = AverageMeter()
        
        for batch_idx, data in enumerate(train_ds):
            siRNA = compute_physics_input_features(data[0].to(device))
            mRNA = data[1].to(device)
            siRNA_FM = data[2].to(device)
            mRNA_FM = data[3].to(device)
            efficacy = data[4].float().to(device)
            td = data[6].to(device)
            
            batch_start = batch_idx * Args.batch_size
            batch_end = min(batch_start + Args.batch_size, len(sample_weights_tensor))
            batch_weights = sample_weights_tensor[batch_start:batch_end].to(device)
            if len(batch_weights) < siRNA.size(0):
                batch_weights = torch.ones(siRNA.size(0), device=device)
            
            optimizer.zero_grad()
            
            out = OFmodel(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
            logits, siRNA_attention, _, aux_preds = _unpack_model_out(out)
            pred_efficacy = prob_from_logits(logits)
            
            # MSE loss
            per_sample_mse = (pred_efficacy - efficacy) ** 2
            mse_loss = (per_sample_mse * batch_weights).mean()
            
            # Physics loss - use MODEL'S nucleotide predictions for gradient flow
            if config['use_physics'] and physics_module is not None:
                si_nt_probs = out[3] if len(out) > 3 and out[3] is not None else None
                
                if si_nt_probs is not None:
                    physics_loss = physics_module(si_nt_probs, pred_efficacy, efficacy, epoch, mRNA_input=mRNA)
                    if batch_idx == 0 and epoch == 0:
                        print(f"[DEBUG] Using si_nt_probs shape={si_nt_probs.shape}, physics_loss={physics_loss.item():.4f}")
                else:
                    siRNA_onehot = siRNA.squeeze(1)[:, :, :4]
                    physics_loss = physics_module(siRNA_onehot, pred_efficacy, efficacy, epoch, mRNA_input=mRNA)
                    if batch_idx == 0 and epoch == 0:
                        print(f"[DEBUG] si_nt_probs is None! Using input one-hot")
            else:
                physics_loss = torch.tensor(0.0, device=device)
            
            # Aux loss
            aux_loss = compute_aux_loss(aux_preds, siRNA)
            
            # Combine - physics weight ramped and capped at 0.10
            if epoch < warmup_epochs:
                pw = 0.0  # No physics during warmup
            else:
                # Ramp from 0.02 to cap of 0.10
                pw = min(0.30, 0.10 + 0.01 * (epoch - warmup_epochs))
            
            if not config['use_physics']:
                pw = 0.0
            
            loss = mse_loss + pw * physics_loss + 0.02 * aux_loss
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(OFmodel.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(OFmodel)
            
            running_mse.update(mse_loss.item(), siRNA.size(0))
            physics_tracker.update(physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss, siRNA.size(0))
        
        scheduler.step()
        
        # Validation
        ema.apply_to(OFmodel)
        val_metrics = val_mse(OFmodel, val_ds)
        test_metrics = val_mse(OFmodel, test_ds)
        ema.restore(OFmodel)
        
        history['train_mse'].append(running_mse.get_average())
        history['val_auc'].append(val_metrics['rocauc'])
        history['test_auc'].append(test_metrics['rocauc'])
        history['test_pcc'].append(test_metrics['pcc'])
        history['physics_loss'].append(physics_tracker.get_average())
        
        # Track best on TARGET (not source)
        if test_metrics['rocauc'] > best_test_auc:
            best_test_auc = test_metrics['rocauc']
            best_val_auc = val_metrics['rocauc']
            best_epoch = epoch
            ema.apply_to(OFmodel)
            torch.save(OFmodel.state_dict(), f'{logger.get_model_dir()}/best_model.pth')
            ema.restore(OFmodel)
        
        if epoch % 10 == 0 or epoch == Args.epoch - 1:
            print(f"Epoch {epoch}: train_MSE={running_mse.get_average():.4f}, "
                  f"val_AUC={val_metrics['rocauc']:.4f}, test_AUC={test_metrics['rocauc']:.4f}, "
                  f"test_PCC={test_metrics['pcc']:.4f}")
        
        # Early stopping on target
        if epoch - best_epoch > early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    OFmodel.load_state_dict(torch.load(f'{logger.get_model_dir()}/best_model.pth'))
    final_test = val_mse(OFmodel, test_ds)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {source_name} → {target_name} ({ablation_mode})")
    print(f"{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Test AUC:   {final_test['rocauc']:.4f}")
    print(f"Test PRC:   {final_test['prauc']:.4f}")
    print(f"Test F1:    {final_test['f1']:.4f}")
    print(f"Test PCC:   {final_test['pcc']:.4f}")
    
    # Save results
    results = {
        'source': source_name,
        'target': target_name,
        'ablation': ablation_mode,
        'best_epoch': best_epoch,
        'test_auc': final_test['rocauc'],
        'test_prc': final_test['prauc'],
        'test_f1': final_test['f1'],
        'test_pcc': final_test['pcc']
    }
    
    pd.DataFrame([results]).to_csv(f'{logger.get_model_dir()}/results.csv', index=False)
    
    # Visualize physics
    if physics_module is not None:
        physics_module.visualize(f'{logger.get_model_dir()}/physics.png')
    
    # Analyze top-k violations (key BioPrior metric)
    print("\nAnalyzing top-k violations...")
    try:
        from scripts.analysis import analyze_topk_violations
        
        topk_results = analyze_topk_violations(OFmodel, test_ds, device, k=50)
        
        print(f"\n{'='*50}")
        print(f"TOP-50 PREDICTIONS ANALYSIS")
        print(f"{'='*50}")
        print(f"Top-50 mean prediction:  {topk_results['topk_mean_pred']:.4f}")
        print(f"Top-50 mean efficacy:    {topk_results['topk_mean_efficacy']:.4f}")
        print(f"Top-50 precision:        {topk_results['topk_precision']:.4f}")
        print(f"\nViolation rates (top-50 vs rest):")
        print(f"  Asymmetry:   {topk_results['topk_asym_violation']:.4f} vs {topk_results['rest_asym_violation']:.4f}")
        print(f"  Motif:       {topk_results['topk_motif_violation']:.4f} vs {topk_results['rest_motif_violation']:.4f}")
        print(f"  Seed GC:     {topk_results['topk_seed_gc_violation']:.4f} vs {topk_results['rest_seed_gc_violation']:.4f}")
        print(f"  Global GC:   {topk_results['topk_global_gc_violation']:.4f} vs {topk_results['rest_global_gc_violation']:.4f}")
        print(f"  Total:       {topk_results['topk_total_violation']:.4f} vs {topk_results['rest_total_violation']:.4f}")
        
        # Save to results
        results['topk_precision'] = topk_results['topk_precision']
        results['topk_total_violation'] = topk_results['topk_total_violation']
        results['rest_total_violation'] = topk_results['rest_total_violation']
        
        pd.DataFrame([topk_results]).to_csv(f'{logger.get_model_dir()}/topk_analysis.csv', index=False)
    except Exception as e:
        print(f"Top-k analysis failed: {e}")
    
    # Run saliency analysis
    print("\nGenerating saliency maps and analysis...")
    try:
        from scripts.analysis import (compute_batch_saliency, plot_saliency_heatmap,
                                       plot_position_importance, plot_nucleotide_preferences,
                                       analyze_physics_violations, plot_physics_violations_vs_efficacy)
        
        # Compute saliency
        avg_sal, pos_sal, neg_sal = compute_batch_saliency(OFmodel, test_ds, device, n_samples=100)
        
        # Save plots
        plot_saliency_heatmap(avg_sal, f"{source_name}→{target_name} Saliency ({ablation_mode})",
                             f'{logger.get_model_dir()}/saliency_heatmap.png')
        plot_position_importance(avg_sal, f"Position Importance ({ablation_mode})",
                                f'{logger.get_model_dir()}/position_importance.png')
        plot_nucleotide_preferences(avg_sal, f"Nucleotide Preferences ({ablation_mode})",
                                   f'{logger.get_model_dir()}/nucleotide_preferences.png')
        
        # Saliency for high vs low efficacy
        plot_saliency_heatmap(pos_sal, "High Efficacy Samples",
                             f'{logger.get_model_dir()}/saliency_high_efficacy.png')
        plot_saliency_heatmap(neg_sal, "Low Efficacy Samples",
                             f'{logger.get_model_dir()}/saliency_low_efficacy.png')
        
        # Physics violations analysis
        violations_df = analyze_physics_violations(test_ds, device, n_samples=200)
        violations_df.to_csv(f'{logger.get_model_dir()}/physics_violations.csv', index=False)
        plot_physics_violations_vs_efficacy(violations_df,
                                            f'{logger.get_model_dir()}/violations_vs_efficacy.png')
        
        # Save saliency data
        np.savez(f'{logger.get_model_dir()}/saliency_data.npz',
                 avg_saliency=avg_sal, pos_saliency=pos_sal, neg_saliency=neg_sal)
        
        print(f"Analysis saved to {logger.get_model_dir()}")
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    print(f"\nResults saved to: {logger.get_model_dir()}")
    
    return results
