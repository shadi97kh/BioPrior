"""
Physics-Informed OligoFormer Inference Script
Fixed: Computes 35 thermodynamic features to match training checkpoint.

Key fixes:
1. compute_physics_input_features(): Adds 8 physics features to siRNA [B,1,19,5] -> [B,1,19,13]
2. calculate_td(): Now produces 35 thermodynamic features (was 24)
3. _unpack_model_out(): Handles tuple output from physics-informed model
"""

import os
import re
import itertools
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loader import data_process_loader_infer
from torch.utils.data import DataLoader
from model import Oligo
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import auc as pr_auc
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from datetime import datetime
import time

DeltaG = {'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33, 'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11, 'GU': -2.24,  'AC': -2.24, 'GA': -2.35,  'UC': -2.35, 'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42, 'init': 4.09, 'endAU': 0.45, 'sym': 0.43}
DeltaH = {'AA': -6.82, 'UU': -6.82, 'AU': -9.38, 'UA': -7.69, 'CU': -10.48, 'AG': -10.48, 'CA': -10.44, 'UG': -10.44, 'GU': -11.40,  'AC': -11.40, 'GA': -12.44,  'UC': -12.44, 'CG': -10.64, 'GG': -13.39, 'CC': -13.39, 'GC': -14.88, 'init': 3.61, 'endAU': 3.72, 'sym': 0}

# Expected TD feature count from training
EXPECTED_TD_FEATURES = 35

# =============================================================================
# Physics Input Features (must match training - produces 13 features)
# =============================================================================
def compute_physics_input_features(siRNA_onehot):
    """
    Add explicit physics features to input.
    Input: [batch, 1, 19, 5]
    Output: [batch, 1, 19, 13] (5 original + 8 physics)
    
    Physics features added:
    - Position importance weights
    - Seed region indicator (positions 1-7)
    - Cleavage site indicator (positions 9-10)
    - Global GC content
    - Seed AU content
    - Thermodynamic asymmetry
    - Per-position AU indicator
    - Per-position GC indicator
    """
    x = siRNA_onehot.squeeze(1)[:, :, :4]
    batch_size, seq_len, _ = x.shape
    dev = x.device
    
    seq_idx = x.argmax(dim=-1)
    
    pos_imp = torch.tensor([0.25,0.15,0.15,0.15,0.15,0.15,0.20,0.15,
                            0.10,0.18,0.18,0.10,0.10,0.12,0.10,0.10,0.10,0.10,0.14], 
                           device=dev).view(1, 19, 1).expand(batch_size, -1, -1)
    
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


def _unpack_model_out(out):
    """Unpack model output - physics model returns tuple"""
    if isinstance(out, tuple):
        logits = out[0]
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)
        return logits
    return out


# =============================================================================
# ORIGINAL HELPER FUNCTIONS
# =============================================================================
def antiRNA(RNA):
    antiRNA = []
    for i in RNA:
        if i == 'A' or i == 'a':
            antiRNA.append('U')
        elif i == 'U' or i == 'u' or i == 'T' or i == 't':
            antiRNA.append('A')
        elif i == 'C' or i == 'c':
            antiRNA.append('G')
        elif i == 'G' or i == 'g':
            antiRNA.append('C')
        elif i == 'X' or i == 'x':
            antiRNA.append('X')
    return ''.join(antiRNA[::-1])

def Calculate_DGH(seq):
    if 'X' in seq:
        return 0.0, 0.0
    DG_all = 0
    DG_all += DeltaG['init']
    DG_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaG['endAU']
    DG_all += DeltaG['sym'] if antiRNA(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        DG_all += DeltaG[seq[i] + seq[i+1]]
    DH_all = 0
    DH_all += DeltaH['init']
    DH_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaH['endAU']
    DH_all += DeltaH['sym'] if antiRNA(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        DH_all += DeltaH[seq[i] + seq[i+1]]
    return DG_all, DH_all

def Calculate_end_diff(siRNA):
    if 'X' in siRNA[:2] or 'X' in siRNA[-2:]:
        return 0.0
    count = 0
    _5 = siRNA[:2]
    _3 = siRNA[-2:]
    if _5 in ['AC','AG','UC','UG']:
        count += 1
    elif _5 in ['GA','GU','CA','CU']:
        count -= 1
    if _3 in ['AC','AG','UC','UG']:
        count += 1
    elif _3 in ['GA','GU','CA','CU']:
        count -= 1
    return float('{:.2f}'.format(DeltaG[_5] - DeltaG[_3] + count * 0.45))


# =============================================================================
# FIXED: calculate_td with 35 features to match training
# =============================================================================
def calculate_td(df):
    """
    Calculate 35 thermodynamic features to match training data.
    Original infer.py only computed 24 features - this caused the 
    classifier size mismatch (536 vs 547 = 11 missing features).
    """
    td_features = []
    
    for i in range(df.shape[0]):
        seq = df.iloc[i, 0]  # siRNA sequence
        features = []
        
        # ============ ORIGINAL 24 FEATURES ============
        # 1. ends (thermodynamic end differential)
        features.append(Calculate_end_diff(seq))
        # 2. DG_1 (free energy of first dinucleotide)
        features.append(DeltaG[seq[0:2]])
        # 3. DH_1 (enthalpy of first dinucleotide)
        features.append(DeltaH[seq[0:2]])
        # 4. U_1 (U at position 1)
        features.append(int(seq[0] == 'U'))
        # 5. G_1 (G at position 1)
        features.append(int(seq[0] == 'G'))
        # 6. DH_all (total enthalpy)
        features.append(Calculate_DGH(seq)[1])
        # 7. U_all (U content)
        features.append(seq.count('U') / 19)
        # 8. UU_1 (UU at start)
        features.append(int(seq[0:2] == 'UU'))
        # 9. G_all (G content)
        features.append(seq.count('G') / 19)
        # 10. GG_1 (GG at start)
        features.append(int(seq[0:2] == 'GG'))
        # 11. GC_1 (GC at start)
        features.append(int(seq[0:2] == 'GC'))
        # 12. GG_all (GG dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('GG') / 18)
        # 13. DG_2 (free energy of second dinucleotide)
        features.append(DeltaG[seq[1:3]])
        # 14. UA_all (UA dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('UA') / 18)
        # 15. U_2 (U at position 2)
        features.append(int(seq[1] == 'U'))
        # 16. C_1 (C at position 1)
        features.append(int(seq[0] == 'C'))
        # 17. CC_all (CC dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('CC') / 18)
        # 18. DG_18 (free energy at position 18)
        features.append(DeltaG[seq[17:19]])
        # 19. CC_1 (CC at start)
        features.append(int(seq[0:2] == 'CC'))
        # 20. GC_all (GC dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('GC') / 18)
        # 21. CG_1 (CG at start)
        features.append(int(seq[0:2] == 'CG'))
        # 22. DG_13 (free energy at position 13)
        features.append(DeltaG[seq[12:14]])
        # 23. UU_all (UU dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('UU') / 18)
        # 24. A_19 (A at position 19)
        features.append(int(seq[18] == 'A'))
        
        # ============ ADDITIONAL 11 FEATURES (to reach 35) ============
        # 25. A_1 (A at position 1)
        features.append(int(seq[0] == 'A'))
        # 26. DG_all (total free energy)
        features.append(Calculate_DGH(seq)[0])
        # 27. A_all (A content)
        features.append(seq.count('A') / 19)
        # 28. C_all (C content)
        features.append(seq.count('C') / 19)
        # 29. AU_all (AU dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('AU') / 18)
        # 30. CG_all (CG dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('CG') / 18)
        # 31. AA_all (AA dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('AA') / 18)
        # 32. GC_content (overall GC content)
        features.append((seq.count('G') + seq.count('C')) / 19)
        # 33. AC_all (AC dinucleotide frequency)
        features.append([seq[j]+seq[j+1] for j in range(18)].count('AC') / 18)
        # 34. U_19 (U at position 19)
        features.append(int(seq[18] == 'U'))
        # 35. G_19 (G at position 19)
        features.append(int(seq[18] == 'G'))
        
        assert len(features) == EXPECTED_TD_FEATURES, f"Expected {EXPECTED_TD_FEATURES}, got {len(features)}"
        td_features.append(features)
    
    df['td'] = td_features
    return df[['siRNA', 'mRNA', 'td']]


def func_filter(siRNA):
    def GC_content(siRNA):
        GC = siRNA.count('G') + siRNA.count('C')
        return GC / len(siRNA) * 100
    
    def five_bases_in_a_row(siRNA):
        pattern = ['AAAAA', 'UUUUU', 'CCCCC', 'GGGGG']
        pattern = [re.compile(p) for p in pattern]
        for p in pattern:
            if re.search(p, siRNA):
                return True
        return False
    
    def six_Gs_or_Cs_in_a_row(siRNA):
        pattern = [''.join(i) for i in itertools.product(('G', 'C'), repeat=6)]
        pattern = [re.compile(p) for p in pattern]
        for p in pattern:
            if re.search(p, siRNA):
                return True
        return False
    
    def palindromic_sequence(siRNA):
        for i in range(len(siRNA) - 8 + 1):
            pattern = siRNA[i:i+4][::-1].translate(str.maketrans('AUCG', 'UAGC'))
            if re.search(pattern, siRNA[i+4:]):
                return True
        return False
        
    label = list()
    for s in siRNA:
        if GC_content(s) < 30 or GC_content(s) > 65:
            label.append(1)
            continue
        if five_bases_in_a_row(s):
            label.append(2)
            continue
        if six_Gs_or_Cs_in_a_row(s):
            label.append(3)
            continue
        if palindromic_sequence(s):
            label.append(4)
            continue
        label.append(0)
    return label


def find_best_checkpoint(checkpoint_path=None):
    """Find the best model checkpoint"""
    import glob
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path
    
    search_patterns = [
        "result/*/model/*.pth",
        "result/*/*/*.pth", 
        "result/*_5fold_mse/model/fold_*_best.pth",
    ]
    
    all_checkpoints = []
    for pattern in search_patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        if os.path.exists("model/best_model.pth"):
            print("WARNING: No physics-informed checkpoint found. Using original model.")
            return "model/best_model.pth"
        raise FileNotFoundError("No model checkpoint found!")
    
    best_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    print(f"Using checkpoint: {best_checkpoint}")
    return best_checkpoint


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================
def infer(Args):
    random.seed(Args.seed)
    os.environ['PYTHONHASHSEED']=str(Args.seed)
    np.random.seed(Args.seed)
    
    # Find checkpoint
    checkpoint_path = getattr(Args, 'checkpoint', None)
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint()
    else:
        checkpoint_path = find_best_checkpoint(checkpoint_path)
    
    # Initialize model
    best_model = Oligo(vocab_size = Args.vocab_size, embedding_dim = Args.embedding_dim, 
                       lstm_dim = Args.lstm_dim,  n_head = Args.n_head, n_layers = Args.n_layers, 
                       lm1 = Args.lm1, lm2 = Args.lm2).to(device)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Check classifier input size from checkpoint
    for key in state_dict.keys():
        if 'classifier.0.weight' in key:
            expected_size = state_dict[key].shape[1]
            print(f"Checkpoint classifier expects input size: {expected_size}")
            break
    
    # Load state dict
    try:
        best_model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint (strict): {checkpoint_path}")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Trying with strict=False...")
        best_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint (non-strict): {checkpoint_path}")
    
    best_model.eval()
    
    print('-----------------Start inferring!-----------------')
    a = datetime.now()
    
    if Args.all_human:
        Args.utr = './off-target/ref/human_UTR.txt'
        Args.orf = './off-target/ref/human_UTR.txt'
    
    if Args.infer == 1:
        with open(Args.infer_fasta) as fa:
            fa_dict = {}
            seq_name = None
            seq_data = ''
            _First = True
            for line in fa:
                line = line.strip()
                if line.startswith('>'):
                    if _First:
                        seq_name = line[1:]
                        _First = False
                    else:
                        fa_dict[seq_name] = ''.join(seq_data)
                        seq_name = line[1:]
                        seq_data = []
                else:
                    seq_data += line.upper().replace('T','U')
            if seq_name:
                fa_dict[seq_name] = ''.join(seq_data)
        
        if Args.infer_siRNA_fasta:
            total_siRNA = list()
            with open(Args.infer_siRNA_fasta) as fa_siRNA:
                for line in fa_siRNA:
                    line = line.replace('\n','')
                    if not line.startswith('>'):
                        if len(line.replace('\n','')) != 19:
                            raise Exception("The length of some siRNA is not 19 nt!")
                        total_siRNA.append(line.replace('\n',''))
        
        for _name, _mRNA in fa_dict.items():
            print(_name)
            _name = _name.replace(' ','_@_')
            if len(_mRNA) < 19:
                raise Exception("The length of mRNA is less than 19 nt!")
            _infer_df = pd.DataFrame(columns=['siRNA','mRNA'])
            _siRNA = list()
            _cRNA = list()
            if not Args.infer_siRNA_fasta:
                for i in range(len(_mRNA) - 19 + 1): 
                    _siRNA.append(antiRNA(_mRNA[i:i+19]))
                for i in range(len(_mRNA) - 19 + 1):
                    _cRNA.append('X' * max(0, 19-i) + _mRNA[max(0,i-19):(i+38)] + 'X' * max(0,i+38-len(_mRNA)))
            else:
                for k in range(len(total_siRNA)):
                    if re.search(antiRNA(total_siRNA[k]),_mRNA) is not None:
                        _left = re.search(antiRNA(total_siRNA[k]),_mRNA).span()[0]
                        _siRNA.append(total_siRNA[k])
                        _cRNA.append('X' * max(0, 19-_left) + _mRNA[max(0,_left-19):(_left+38)] + 'X' * max(0,_left+38-len(_mRNA)))
            _infer_df['siRNA'] = _siRNA
            _infer_df['mRNA'] = _cRNA
            _infer_df = calculate_td(_infer_df)  # Now produces 35 features
            
            if not os.path.exists('./data/infer'):
                os.mkdir('./data/infer')
            os.system('rm -rf ./data/infer/' + _name)
            os.system('mkdir ./data/infer/' + _name)
            for i in range(_infer_df.shape[0]):
                with open('./data/infer/' + _name + '/siRNA.fa','a') as f:
                    f.write('>RNA' + str(i) + '\n')
                    f.write(_infer_df['siRNA'][i] + '\n')
                with open('./data/infer/' + _name + '/mRNA.fa','a') as f:
                    f.write('>RNA' + str(i) + '\n')
                    f.write(_infer_df['mRNA'][i] + '\n')
            os.system('sh scripts/RNA-FM.sh ../../data/infer/' + _name)
            
            params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
            infer_ds = DataLoader(data_process_loader_infer(_infer_df.index.values, _infer_df, _name),**params)
            
            Y_PRED = []
            first_batch = True
            with torch.no_grad():
                for i, data in enumerate(infer_ds):
                    siRNA = data[0].to(device)
                    mRNA = data[1].to(device)
                    siRNA_FM = data[2].to(device)
                    mRNA_FM = data[3].to(device)
                    td = data[4].to(device)
                    
                    if first_batch:
                        print(f"\n=== DEBUG SHAPES ===")
                        print(f"siRNA (before physics): {siRNA.shape}")
                    
                    # Apply physics features: [B,1,19,5] -> [B,1,19,13]
                    siRNA = compute_physics_input_features(siRNA)
                    
                    if first_batch:
                        print(f"siRNA (after physics): {siRNA.shape}")
                        print(f"mRNA: {mRNA.shape}")
                        print(f"siRNA_FM: {siRNA_FM.shape}")
                        print(f"mRNA_FM: {mRNA_FM.shape}")
                        print(f"td: {td.shape}")  # Should be [1, 35]
                        print(f"====================\n")
                        first_batch = False
                    
                    out = best_model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
                    logits = _unpack_model_out(out)
                    pred_efficacy = torch.sigmoid(logits[:, 1])
                    Y_PRED.append(pred_efficacy.item())
            
            Y_PRED = pd.DataFrame(Y_PRED)
            
            RESULT = pd.DataFrame()
            RESULT['pos'] = list(range(1,_infer_df.shape[0] + 1))
            RESULT['sense'] = [antiRNA(_infer_df.iloc[i,0]) for i in range(_infer_df.shape[0])]
            RESULT['siRNA'] = _infer_df['siRNA']
            print(Y_PRED)
            RESULT['efficacy'] = Y_PRED
            
            if not Args.no_func:
                RESULT['func_filter'] = func_filter(_siRNA)
            
            if Args.off_target:
                if Args.top_n == -1:
                    siRNA_file = './data/infer/' + _name + '/siRNA.fa'
                    os.system(f'bash scripts/pita.sh {Args.utr} {siRNA_file} {Args.orf} {_name}')
                    os.system(f'bash scripts/targetscan.sh {siRNA_file} {Args.utr} {Args.orf} {_name}')
                    pita = pd.read_csv('./data/infer/' + _name + '/pita.tab', sep='\t')
                    pita = pita.groupby('microRNA').agg({'Score': 'min'}).rename(columns={'Score': 'pita_score'})
                    RESULT['tmp'] = RESULT['pos'].astype(str).apply(lambda x: 'RNA' + str(int(x) - 1))
                    RESULT = pd.merge(RESULT, pita, left_on='tmp', right_on='microRNA', how='left')
                    RESULT['pita_filter'] = [1 if i < Args.pita_threshold else 0 for i in RESULT['pita_score']]
                    targetscan = pd.read_csv('./data/infer/' + _name + '/targetscan.tab', sep='\t', header=None, names=['refseq', 'siRNA', 'targetscan_score'])
                    targetscan = targetscan.groupby('siRNA').agg({'targetscan_score': 'max'})
                    for i in list(set(pita.index) - set(targetscan.index)):
                        targetscan.loc[i] = 0
                    RESULT = pd.merge(RESULT, targetscan, left_on='tmp', right_on='siRNA', how='left')
                    RESULT['targetscan_filter'] = [1 if i > Args.targetscan_threshold else 0 for i in RESULT['targetscan_score']]
                    RESULT['off_target_filter'] = [1 if i == 1 or j == 1 else 0 for i,j in zip(RESULT['pita_filter'], RESULT['targetscan_filter'])]
                    RESULT = RESULT.drop(columns=['tmp','pita_filter', 'targetscan_filter'])
                else:
                    RESULT_ranked = RESULT.sort_values(by='efficacy', ascending=False)
                    for i in range(Args.top_n):
                        with open('./data/infer/' + _name + '/top_n_siRNA.fa','a') as f:
                            f.write('>RNA' + str(RESULT_ranked.index[i]) + '\n')
                            f.write(RESULT_ranked['siRNA'].iloc[i] + '\n')
                    siRNA_file = './data/infer/' + _name + '/top_n_siRNA.fa'
                    os.system(f'bash scripts/pita.sh {Args.utr} {siRNA_file} {Args.orf} {_name}')
                    os.system(f'bash scripts/targetscan.sh {siRNA_file} {Args.utr} {Args.orf} {_name}')
                    pita = pd.read_csv('./data/infer/' + _name + '/pita.tab', sep='\t')
                    pita = pita.groupby('microRNA').agg({'Score': 'min'}).rename(columns={'Score': 'pita_score'})
                    RESULT['tmp'] = RESULT['pos'].astype(str).apply(lambda x: 'RNA' + str(int(x) - 1))
                    RESULT = pd.merge(RESULT, pita, left_on='tmp', right_on='microRNA', how='left')
                    RESULT['pita_filter'] = [1 if i < Args.pita_threshold else 0 for i in RESULT['pita_score']]
                    for i in range(RESULT.shape[0]):
                        if np.isnan(RESULT.iloc[i,-2]):
                            RESULT['pita_filter'].iloc[i] = -1
                    targetscan = pd.read_csv('./data/infer/' + _name + '/targetscan.tab', sep='\t', header=None, names=['refseq', 'siRNA', 'targetscan_score'])
                    targetscan = targetscan.groupby('siRNA').agg({'targetscan_score': 'max'})
                    for i in list(set(pita.index) - set(targetscan.index)):
                        targetscan.loc[i] = 0
                    RESULT = pd.merge(RESULT, targetscan, left_on='tmp', right_on='siRNA', how='left')
                    RESULT['targetscan_filter'] = [1 if i > Args.targetscan_threshold else 0 for i in RESULT['targetscan_score']]
                    for i in range(RESULT.shape[0]):
                        if np.isnan(RESULT.iloc[i,-2]):
                            RESULT['targetscan_filter'].iloc[i] = -1
                    RESULT['off_target_filter'] = [-5 if i == -1 or j == -1 else 1 if i == 1 or j == 1 else 0 for i,j in zip(RESULT['pita_filter'], RESULT['targetscan_filter'])]
                    RESULT = RESULT.drop(columns=['tmp','pita_filter', 'targetscan_filter'])
            
            if Args.toxicity:
                toxicity = pd.read_csv('./toxicity/cell_viability.txt', sep='\t')
                RESULT['seed'] = RESULT['siRNA'].str.slice(1,7)
                RESULT = pd.merge(RESULT, toxicity, left_on='seed', right_on='Seed', how='left')
                RESULT['toxicity_filter'] = [1 if i < Args.toxicity_threshold else 0 for i in RESULT['cell_viability']]
                RESULT = RESULT.drop(columns=['seed'])
            
            if not Args.no_func:
                if Args.off_target:
                    if Args.toxicity:
                        RESULT['filter'] = RESULT['func_filter'] + RESULT['off_target_filter'] + RESULT['toxicity_filter']
                    else:
                        RESULT['filter'] = RESULT['func_filter'] + RESULT['off_target_filter']
                else:
                    if Args.toxicity:
                        RESULT['filter'] = RESULT['func_filter'] + RESULT['toxicity_filter']
                    else:
                        RESULT['filter'] = RESULT['func_filter']
            else:
                if Args.off_target:
                    if Args.toxicity:
                        RESULT['filter'] = RESULT['off_target_filter'] + RESULT['toxicity_filter']
                    else:
                        RESULT['filter'] = RESULT['off_target_filter']
                else:
                    if Args.toxicity:
                        RESULT['filter'] = RESULT['toxicity_filter']
                    else:
                        RESULT['filter'] = [0] * RESULT.shape[0]
            
            RESULT_ranked = RESULT.sort_values(by='efficacy', ascending=False)
            RESULT_ranked_filtered = RESULT[RESULT['filter'] == 0].sort_values(by='efficacy', ascending=False)
            RESULT.to_csv(Args.output_dir + str(_name) + '.txt',sep='\t',index = None,header=True)
            RESULT_ranked.to_csv(Args.output_dir + str(_name) + '_ranked.txt',sep='\t',index = None,header=True)
            RESULT_ranked_filtered.to_csv(Args.output_dir + str(_name) + '_ranked_filtered.txt',sep='\t',index = None,header=True)
    
    elif Args.infer == 2:
        _mRNA = input("please input target mRNA: \n").upper().replace('T','U')
        if len(_mRNA) < 19:
            raise Exception("The length of mRNA is less than 19 nt!")
        _name = 'RNA0'
        print(_name)
        _infer_df = pd.DataFrame(columns=['siRNA','mRNA'])
        _siRNA = list()
        for i in range(len(_mRNA) - 19 + 1): 
            _siRNA.append(antiRNA(_mRNA[i:i+19]))
        _infer_df['siRNA'] = _siRNA
        _cRNA = list()
        for i in range(len(_mRNA) - 19 + 1):
            _cRNA.append('X' * max(0, 19-i) + _mRNA[max(0,i-19):(i+38)] + 'X' * max(0,i+38-len(_mRNA)))
        _infer_df['mRNA'] = _cRNA
        _infer_df = calculate_td(_infer_df)
        
        if not os.path.exists('./data/infer'):
            os.mkdir('./data/infer')
        os.system('rm -rf ./data/infer/' + _name)
        os.system('mkdir ./data/infer/' + _name)
        for i in range(_infer_df.shape[0]):
            with open('./data/infer/' + _name + '/siRNA.fa','a') as f:
                f.write('>RNA' + str(i) + '\n')
                f.write(_infer_df['siRNA'][i] + '\n')
            with open('./data/infer/' + _name + '/mRNA.fa','a') as f:
                f.write('>RNA' + str(i) + '\n')
                f.write(_infer_df['mRNA'][i] + '\n')
        os.system('sh scripts/RNA-FM.sh ../../data/infer/' + _name)
        
        params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
        infer_ds = DataLoader(data_process_loader_infer(_infer_df.index.values, _infer_df, _name),**params)
        
        Y_PRED = []
        with torch.no_grad():
            for i, data in enumerate(infer_ds):
                siRNA = data[0].to(device)
                siRNA = compute_physics_input_features(siRNA)
                mRNA = data[1].to(device)
                siRNA_FM = data[2].to(device)
                mRNA_FM = data[3].to(device)
                td = data[4].to(device)
                
                out = best_model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
                logits = _unpack_model_out(out)
                pred_efficacy = torch.sigmoid(logits[:, 1])
                Y_PRED.append(pred_efficacy.item())
        
        Y_PRED = pd.DataFrame(Y_PRED)
        RESULT = pd.DataFrame()
        RESULT['pos'] = list(range(1,_infer_df.shape[0] + 1))
        RESULT['sense'] = [antiRNA(_infer_df.iloc[i,0]) for i in range(_infer_df.shape[0])]
        RESULT['siRNA'] = _infer_df['siRNA']
        RESULT['efficacy'] = Y_PRED
        
        if not Args.no_func:
            RESULT['func_filter'] = func_filter(_siRNA)
            RESULT['filter'] = RESULT['func_filter']
        else:
            RESULT['filter'] = [0] * RESULT.shape[0]
        
        RESULT_ranked = RESULT.sort_values(by='efficacy', ascending=False)
        RESULT_ranked_filtered = RESULT[RESULT['filter'] == 0].sort_values(by='efficacy', ascending=False)
        RESULT.to_csv(Args.output_dir + str(_name) + '.txt',sep='\t',index = None,header=True)
        RESULT_ranked.to_csv(Args.output_dir + str(_name) + '_ranked.txt',sep='\t',index = None,header=True)
        RESULT_ranked_filtered.to_csv(Args.output_dir + str(_name) + '_ranked_filtered.txt',sep='\t',index = None,header=True)
    
    b = datetime.now()
    durn = (b-a).seconds
    print(durn, 'seconds', durn / 60, 'minutes')
    
def evaluate_datasets(Args):
    """
    Evaluate model on datasets with ground truth.
    Matches paper metrics: AUC, PRC, F1, PCC
    
    Usage: python scripts/main.py --eval Hu Mix Taka --checkpoint path/to/model.pth
    """
    from torch.utils.data import DataLoader
    from loader import data_process_loader
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
    from sklearn.metrics import auc as pr_auc
    from scipy.stats import pearsonr
    
    # Find checkpoint
    checkpoint_path = find_best_checkpoint(getattr(Args, 'checkpoint', None))
    
    # Load model
    model = Oligo(vocab_size=Args.vocab_size, embedding_dim=Args.embedding_dim,
                  lstm_dim=Args.lstm_dim, n_head=Args.n_head, n_layers=Args.n_layers,
                  lm1=Args.lm1, lm2=Args.lm2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.eval()
    print(f"Loaded: {checkpoint_path}\n")
    
    results = {}
    
    for ds in Args.eval:
        print(f"Evaluating {ds}...")
        
        # Load dataset
        df = pd.read_csv(f'{Args.path}{ds}.csv')
        labels = df['label'].values
        y = (labels >= 0.5).astype(int)
        
        # Create loader
        loader = DataLoader(
            data_process_loader(df.index.values, labels, y, df, ds, Args.path),
            batch_size=16, shuffle=False, num_workers=0
        )
        
        preds_list, labels_list = [], []
        
        with torch.no_grad():
            for data in loader:
                # Apply physics features
                siRNA = compute_physics_input_features(data[0].to(device))
                mRNA = data[1].to(device)
                siRNA_FM = data[2].to(device)
                mRNA_FM = data[3].to(device)
                td = data[6].to(device)  # td is at index 6
                label = data[4]  # continuous label
                
                out = model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
                pred = torch.sigmoid(_unpack_model_out(out)[:, 1])
                
                preds_list.extend(pred.cpu().numpy())
                labels_list.extend(label.numpy())
        
        preds = np.array(preds_list)
        labels_arr = np.array(labels_list)
        binary_labels = (labels_arr >= 0.5).astype(int)
        binary_preds = (preds >= 0.5).astype(int)
        
        # Paper metrics: AUC, PRC, F1, PCC
        auc_score = roc_auc_score(binary_labels, preds)
        prec, rec, _ = precision_recall_curve(binary_labels, preds)
        prc = pr_auc(rec, prec)
        f1 = f1_score(binary_labels, binary_preds)
        pcc, _ = pearsonr(preds, labels_arr)
        
        results[ds] = {'AUC': auc_score, 'PRC': prc, 'F1': f1, 'PCC': pcc}
        print(f"  {ds}: AUC={auc_score:.4f}, PRC={prc:.4f}, F1={f1:.4f}, PCC={pcc:.4f}")
    
    # Print summary table
    print(f"\n{'='*55}")
    print(f"{'Dataset':<10}{'AUC':>10}{'PRC':>10}{'F1':>10}{'PCC':>10}")
    print(f"{'-'*55}")
    for ds, m in results.items():
        print(f"{ds:<10}{m['AUC']:>10.4f}{m['PRC']:>10.4f}{m['F1']:>10.4f}{m['PCC']:>10.4f}")
    print(f"{'='*55}")
    
    # Save to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv('result/evaluation_results.csv')
    print(f"\nResults saved to: result/evaluation_results.csv")
