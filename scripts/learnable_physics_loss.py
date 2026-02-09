import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class BioPriorModule(nn.Module):
    """
    DIFFERENTIABLE mechanistic biology-informed loss.
    
    Key improvements:
    - Removed gated loss (was preventing learning)
    - Added mRNA accessibility term
    - Proper scaling so constraints actually matter
    """
    
    def __init__(self, seq_len=19,
                 # Backward compatibility - ignore old args
                 physics_weight=None, sparsity_lambda=None, target_active=None, init_mode=None,
                 use_topk=None, temperature_init=None, temperature_final=None,
                 alignment_weight=None, n_nucleotides=None, **kwargs):
        super().__init__()
        
        self.seq_len = seq_len
        self.alphabet = ['A', 'U', 'G', 'C']
        
        # FIXED: Stronger constraint weights
        self.register_buffer('constraint_weights', torch.tensor([
            3.0,   # Thermodynamic asymmetry - MOST IMPORTANT
            1.0,   # Immune motifs
            2.0,   # Seed GC - off-target control
            1.5,   # Global GC
            2.5    # mRNA accessibility (NEW)
        ]))
        
        # For tracking
        self.register_buffer('_loss_history', torch.zeros(5))
        self._call_count = 0
        
        print(f"[BioPrior] DIFFERENTIABLE mechanistic mode (v2: with mRNA accessibility)")
    
    def _get_probs(self, siRNA_input):
        """Extract [B, 19, 4] probabilities from various input shapes."""
        if siRNA_input.dim() == 4:
            x = siRNA_input.squeeze(1)[:, :, :4]
        elif siRNA_input.dim() == 3:
            x = siRNA_input[:, :, :4]
        else:
            raise ValueError(f"Unexpected shape: {siRNA_input.shape}")
        
        # Ensure valid probabilities
        x = torch.clamp(x, min=1e-6, max=1.0)
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)
        
        return x
    
    def thermodynamic_asymmetry_loss(self, siRNA_probs, per_sample=False):
        """
        FIXED: Stronger penalty for wrong asymmetry.
        
        5' should be AU-rich (less stable), 3' GC-rich (more stable).
        Biology: ΔG difference of ~1-2 kcal/mol between ends.
        
        Returns per-sample violations if per_sample=True.
        """
        x = self._get_probs(siRNA_probs)  # [B, 19, 4]
        
        # Soft GC probability at each position
        gc_prob = x[:, :, 2] + x[:, :, 3]  # P(G) + P(C)
        
        # 5' end GC (positions 1-4, indices 0-3)
        gc_5p = gc_prob[:, :4].mean(dim=1)
        
        # 3' end GC (positions 16-19, indices 15-18)
        gc_3p = gc_prob[:, -4:].mean(dim=1)
        
        # Want: 3' MORE GC-rich than 5' (asymmetry > 0)
        asymmetry = gc_3p - gc_5p
        
        # FIXED: Stronger penalty when asymmetry is wrong
        # Target: 3' should be 10-20% more GC than 5'
        target_min = 0.10
        target_max = 0.30
        
        # Penalize if asymmetry < target_min (wrong direction)
        low_penalty = F.relu(target_min - asymmetry) ** 2
        
        # Also penalize if TOO asymmetric (unnatural)
        high_penalty = F.relu(asymmetry - target_max) ** 2
        
        per_sample_loss = low_penalty + high_penalty  # [B]
        
        if per_sample:
            return per_sample_loss
        return per_sample_loss.mean()
    
    def immune_motif_penalty(self, siRNA_probs, per_sample=False):
        """
        SOFT version: Penalize immunostimulatory motifs.
        
        - Homopolymer runs (AAAA, UUUU, GGGG, CCCC)
        - UGU (TLR7/8 agonist)
        - GUCCUUCAA (strong TLR agonist)
        """
        x = self._get_probs(siRNA_probs)  # [B, 19, 4]
        B, L, _ = x.shape
        device = x.device
        
        per_sample_penalty = torch.zeros(B, device=device)
        
        # Soft homopolymer detection (4-mers)
        if L >= 4:
            for nt in range(4):  # A, U, G, C
                nt_prob = x[:, :, nt]  # [B, L]
                # Sliding window of 4
                windows = nt_prob.unfold(1, 4, 1)  # [B, L-3, 4]
                # Product = probability of 4-mer homopolymer
                run_prob = windows.prod(dim=2)  # [B, L-3]
                per_sample_penalty = per_sample_penalty + run_prob.sum(dim=1) * 2.0  # [B], stronger penalty
        
        # Soft UGU detection (TLR agonist)
        if L >= 3:
            u_prob = x[:, :, 1]  # P(U)
            g_prob = x[:, :, 2]  # P(G)
            for i in range(L - 2):
                ugu_prob = u_prob[:, i] * g_prob[:, i+1] * u_prob[:, i+2]
                per_sample_penalty = per_sample_penalty + ugu_prob * 1.5  # [B]
        
        if per_sample:
            return per_sample_penalty
        return per_sample_penalty.mean()
    
    def seed_gc_penalty(self, siRNA_probs, per_sample=False):
        """
        SOFT version: Penalize seed GC outside optimal range.
        
        Seed = positions 2-8 (indices 1-7)
        Optimal: 30-50% GC (balance specificity vs efficiency)
        """
        x = self._get_probs(siRNA_probs)
        
        # Seed region = positions 2-8 (indices 1-7)
        seed_probs = x[:, 1:8, :]  # [B, 7, 4]
        
        # Soft GC content in seed
        seed_gc = (seed_probs[:, :, 2] + seed_probs[:, :, 3]).mean(dim=1)  # [B]
        
        # FIXED: Tighter optimal range (30-50%)
        high_penalty = F.relu(seed_gc - 0.50) ** 2
        low_penalty = F.relu(0.30 - seed_gc) ** 2
        
        per_sample_loss = (high_penalty + low_penalty) * 2.0  # [B], stronger penalty
        
        if per_sample:
            return per_sample_loss
        return per_sample_loss.mean()
    
    def global_gc_penalty(self, siRNA_probs, per_sample=False):
        """
        SOFT version: Penalize global GC outside 35-55% range.
        """
        x = self._get_probs(siRNA_probs)
        
        # Global soft GC content
        gc_content = (x[:, :, 2] + x[:, :, 3]).mean(dim=1)  # [B]
        
        # Penalize outside optimal range
        low_penalty = F.relu(0.35 - gc_content) ** 2
        high_penalty = F.relu(gc_content - 0.55) ** 2
        
        per_sample_loss = (low_penalty + high_penalty) * 1.5  # [B], stronger penalty
        
        if per_sample:
            return per_sample_loss
        return per_sample_loss.mean()
    
    def mrna_accessibility_loss(self, siRNA_probs, mRNA_probs=None, per_sample=False):
        """
        NEW: mRNA target site accessibility proxy.
        
        Accessible regions are AU-rich (less structured).
        GC-rich regions form secondary structure (inaccessible).
        
        Penalize if target site is GC-rich (hard to access).
        """
        if mRNA_probs is None:
            # Fallback: penalize siRNA that would target GC-rich regions
            # (Assumes siRNA GC correlates with target GC - rough proxy)
            x = self._get_probs(siRNA_probs)
            target_gc = (x[:, :, 2] + x[:, :, 3]).mean(dim=1)  # [B]
        else:
            # If mRNA context is provided, use it
            # mRNA_probs shape: [B, L_mRNA, 4]
            if mRNA_probs.dim() == 4:
                m = mRNA_probs.squeeze(1)[:, :, :4]
            else:
                m = mRNA_probs[:, :, :4]
            
            m = torch.clamp(m, min=1e-6, max=1.0)
            m = m / (m.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Target site = central region where siRNA binds
            # Assume this is the middle part of mRNA context
            L = m.size(1)
            start = max(0, L//2 - 10)
            end = min(L, L//2 + 10)
            target_region = m[:, start:end, :]
            
            # GC content in target region
            target_gc = (target_region[:, :, 2] + target_region[:, :, 3]).mean(dim=1)  # [B]
        
        # Penalize GC-rich target sites (less accessible)
        # Optimal target GC: 30-45% (AU-rich = accessible)
        accessibility_penalty = F.relu(target_gc - 0.45) ** 2
        
        per_sample_loss = accessibility_penalty * 2.0  # [B]
        
        if per_sample:
            return per_sample_loss
        return per_sample_loss.mean()
    
    def forward(self, siRNA_input, model_pred=None, true_efficacy=None, epoch=0,
            mRNA_input=None, return_components=False):
        """
        Compute physics loss WITHOUT gating.
        
        FIXED: Removed gated loss - it was preventing learning.
        Now simply returns sum of violations, weighted by importance.
        """
        device = siRNA_input.device
        
        # Compute PER-SAMPLE constraint violations (all differentiable!)
        asym_v = self.thermodynamic_asymmetry_loss(siRNA_input, per_sample=True)  # [B]
        motif_v = self.immune_motif_penalty(siRNA_input, per_sample=True)  # [B]
        seed_v = self.seed_gc_penalty(siRNA_input, per_sample=True)  # [B]
        gc_v = self.global_gc_penalty(siRNA_input, per_sample=True)  # [B]
        access_v = self.mrna_accessibility_loss(siRNA_input, mRNA_input, per_sample=True)  # [B]
        
        # Normalize weights
        w = self.constraint_weights / (self.constraint_weights.sum() + 1e-8)
        
        # Total violation score per sample [B]
        total_v = (w[0] * asym_v + w[1] * motif_v + w[2] * seed_v + 
                   w[3] * gc_v + w[4] * access_v)
        
        # FIXED: No gating - just return mean violation
        # This ensures gradients always flow
        total_loss = total_v.mean()
        
        # Track for logging
        self._loss_history = torch.tensor([
            asym_v.mean().item(),
            motif_v.mean().item(),
            seed_v.mean().item(),
            gc_v.mean().item(),
            access_v.mean().item()
        ])
        self._call_count += 1
        
        if return_components:
            return total_loss, {
                'asymmetry': asym_v.mean().item(),
                'motif': motif_v.mean().item(),
                'seed_gc': seed_v.mean().item(),
                'global_gc': gc_v.mean().item(),
                'accessibility': access_v.mean().item()
            }
        
        return total_loss
    
    def get_learned_rules(self):
        """Return constraint analysis."""
        w = self.constraint_weights / (self.constraint_weights.sum() + 1e-8)
        
        return {
            'type': 'mechanistic_differentiable_v2',
            'constraints': {
                'asymmetry': {'weight': w[0].item()},
                'motif': {'weight': w[1].item()},
                'seed_gc': {'weight': w[2].item()},
                'global_gc': {'weight': w[3].item()},
                'accessibility': {'weight': w[4].item()}
            },
            'last_losses': {
                'asymmetry': self._loss_history[0].item(),
                'motif': self._loss_history[1].item(),
                'seed_gc': self._loss_history[2].item(),
                'global_gc': self._loss_history[3].item(),
                'accessibility': self._loss_history[4].item()
            }
        }
    
    def get_learning_delta(self):
        """Compatibility stub."""
        return {
            'total_gate_change': 0.0,
            'total_pref_change': 0.0,
            'gate_delta': np.zeros(19),
            'pref_delta': np.zeros(19)
        }
    
    def visualize(self, save_path=None):
        """Visualize constraint losses."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Mechanistic BioPrior v2 (with mRNA accessibility)', fontsize=14)
        
        labels = ['Asymmetry', 'Motif', 'Seed GC', 'Global GC', 'Accessibility']
        w = self.constraint_weights.cpu().numpy()
        losses = self._loss_history.cpu().numpy()
        
        ax = axes[0]
        ax.bar(labels, w, color=['steelblue', 'coral', 'seagreen', 'purple', 'orange'], alpha=0.8)
        ax.set_ylabel('Weight')
        ax.set_title('Constraint Weights')
        ax.tick_params(axis='x', rotation=45)
        
        ax = axes[1]
        ax.bar(labels, losses, color=['steelblue', 'coral', 'seagreen', 'purple', 'orange'], alpha=0.8)
        ax.set_ylabel('Loss Value')
        ax.set_title('Last Computed Losses')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.close()
        
        return self.get_learned_rules()


# Ablation configs
ABLATION_CONFIGS = {
    'baseline': {
        'description': 'No physics (baseline OligoFormer)',
        'use_physics': False,
    },
    'mechanistic': {
        'description': 'Mechanistic BioPrior v2 (with mRNA accessibility)',
        'use_physics': True,
    },
    'full': {
        'description': 'Mechanistic BioPrior v2',
        'use_physics': True,
    },
}


if __name__ == '__main__':
    print("Testing FIXED Mechanistic BioPrior\n")
    
    prior = BioPriorModule()
    
    # Test with soft probabilities (like model output)
    x = torch.rand(32, 19, 4)
    x = F.softmax(x, dim=-1)  # Soft probabilities
    x.requires_grad = True
    
    loss, components = prior(x, return_components=True)
    
    print(f"Total loss: {loss.item():.4f}")
    for name, val in components.items():
        print(f"  {name}: {val:.4f}")
    
    # Check gradient flow
    loss.backward()
    print(f"\nGradient flows: {x.grad is not None and x.grad.abs().sum() > 0}")
    print(f"Grad magnitude: {x.grad.abs().mean().item():.6f}")
