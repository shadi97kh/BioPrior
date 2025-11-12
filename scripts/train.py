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
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score,roc_curve,auc,precision_recall_curve,matthews_corrcoef
from metrics import  sensitivity, specificity
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib as mpl
from logger import TrainLogger
from itertools import cycle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calculate_physics_loss(siRNA_batch, epoch=0):
    """Enhanced physics loss with seed region and motif penalties"""
    physics_penalty = 0.0
    batch_size = siRNA_batch.shape[0]
    
    for i in range(batch_size):
        seq_onehot = siRNA_batch[i, 0, :, :4]
        seq = []
        for pos in range(19):
            nt_idx = torch.argmax(seq_onehot[pos]).item()
            seq.append(['A', 'U', 'C', 'G'][nt_idx])
        
        seq_str = ''.join(seq)
        
        # Original position penalties
        if seq[0] != 'U':
            physics_penalty += 0.25 if seq[0] == 'G' else 0.15
        if seq[6] != 'A':
            physics_penalty += 0.12
        if seq[18] != 'C':
            physics_penalty += 0.14
            
        # NEW: Seed region penalties (positions 2-8 - CRITICAL for siRNA)
        seed_region = seq_str[1:8]
        seed_au_content = sum(1 for nt in seed_region if nt in 'AU') / 7
        if seed_au_content < 0.3:  # Seed should have AU content
            physics_penalty += 0.20
        if seq[1] != 'U':  # Position 2 preference
            physics_penalty += 0.08
        if seq[7] != 'A':  # Position 8 preference
            physics_penalty += 0.08
            
        # NEW: Cleavage site (positions 10-11)
        if seq[9] != 'U' or seq[10] != 'A':  # UA at cleavage site
            physics_penalty += 0.10
            
        # NEW: Toxic motif penalties
        if 'GGGG' in seq_str:
            physics_penalty += 0.30  # Heavy penalty for aggregation
        if 'CCCC' in seq_str:
            physics_penalty += 0.30
        if 'AAAAA' in seq_str or 'UUUUU' in seq_str:
            physics_penalty += 0.20  # 5+ repeats are problematic
            
        # NEW: Thermodynamic asymmetry enforcement
        gc_5prime = sum(1 for nt in seq[:4] if nt in 'GC')
        gc_3prime = sum(1 for nt in seq[-4:] if nt in 'GC')
        if gc_5prime > gc_3prime:  # 5' should be less stable
            physics_penalty += 0.25
        
        # NEW: Middle region stability (positions 9-14)
        middle_gc = sum(1 for nt in seq[8:14] if nt in 'GC')
        if middle_gc < 2:  # Too unstable in middle
            physics_penalty += 0.15
    
    # Progressive weight increase with higher max
    physics_weight = min(0.3, 0.1 + 0.002 * epoch)  # Increased from 0.3 to 0.5 max
    return (physics_penalty / batch_size) * physics_weight


def find_metrics_best_for_shuffle(label, prob, cut_spe=0.95):
    fpr, tpr, _ = roc_curve(label, prob)
    a = 1 - fpr
    b = tpr
    Sensitivity = b
    Specificity = a
    Sensitivity_ = Sensitivity[Specificity >= cut_spe]
    if (len(Sensitivity_) == 1) & (Sensitivity_[0] == 0):
        Sensitivity_best = ((Sensitivity[1] - Sensitivity[0]) / (Specificity[1] - Specificity[0])) * cut_spe + Sensitivity[1] - ((Sensitivity[1] - Sensitivity[0]) / (Specificity[1] - Specificity[0])) *  Specificity[1]
    else:
        Sensitivity_best = np.max(Sensitivity_)
    return Sensitivity_best, Sensitivity, Specificity

def get_kfold_data_2(i, datasets, k=5, v=1):
    datasets = shuffle_dataset(datasets, 42).reset_index(drop=True)
    v = v * 10
    if k<5:
        fold_size = len(datasets) // 5
    else:
        fold_size = len(datasets) // k

    test_start = i * fold_size

    if i != k - 1 and i != 0:
        test_end = (i + 1) * fold_size
        TestSet = datasets[test_start:test_end]
        TrainSet = pd.concat([datasets[0:test_start], datasets[test_end:]])

    elif i == 0:
        test_end = fold_size
        TestSet = datasets[test_start:test_end]
        TrainSet = datasets[test_end:]

    else:
        TestSet = datasets[test_start:]
        TrainSet = datasets[0:test_start]

    return TrainSet.reset_index(drop=True), TestSet.reset_index(drop=True)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    dataset = shuffle(dataset)
    return dataset

def write_pkl(pkl_data,pkl_name):
	pkl_file = open(pkl_name, "wb")
	pkl.dump(pkl_data, pkl_file)
	pkl_file.close()

def plotPRC(model,X,Y,name):
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	y_pred = model.predict(X)[:,1]
	precision, recall, threshold = precision_recall_curve(Y[:,1],y_pred)
	prc = auc(recall, precision)
	plt.plot(recall, precision,label='OligoFormer (PRC: %s \u00B1 0.001)' % (np.round(prc, 3)))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='best')
	plt.savefig(name)
	Y_TRUE = pd.DataFrame(Y)
	Y_PRED = pd.DataFrame(model.predict(X)[:,1])
	with open(name.split('PRC')[0] + 'test_prediction.txt', 'w') as f:
		for i in range(Y_TRUE.shape[0]):
			f.write(str(Y_TRUE.iloc[i,1]) + " " + str(Y_PRED.iloc[i,0]) + '\n')

def plotAUC(model,X,Y,name):
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	y_pred = model.predict(X)[:,1]
	fpr, tpr, threshold = roc_curve(Y[:,1],y_pred)
	roc = auc(fpr, tpr)
	plt.plot(fpr, tpr,label='OligoFormer (AUC: %s \u00B1 0.001)' % (np.round(roc, 3)))
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.legend(loc='best')
	plt.savefig(name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg

def val(model, criterion, dataloader):
	running_loss = AverageMeter()
	pred_list = []
	pred_cls_list = []
	label_list = []
	for i, data in enumerate(dataloader):
		siRNA = data[0].to(device)
		mRNA = data[1].to(device)
		siRNA_FM = data[2].to(device)
		mRNA_FM = data[3].to(device)
		label = data[4].to(device)
		td = data[6].to(device)
		pred,_,_ = model(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
		loss = criterion(pred[:,1],label.float())
		label = data[5]
		pred_cls = torch.argmax(pred, dim=-1)
		pred_prob = F.softmax(pred, dim=-1) # pred_prob = pred #
		pred_prob, indices = torch.max(pred_prob, dim=-1)
		pred_prob[indices == 0] = 1. - pred_prob[indices == 0]
		pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
		pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
		label_list.append(label)
		running_loss.update(loss, label.shape[0])
	pred = np.concatenate(pred_list, axis=0)
	pred_cls = np.concatenate(pred_cls_list, axis=0)
	label = np.concatenate(label_list, axis=0)
	acc = accuracy_score(label, pred_cls)
	sen = sensitivity(label,pred_cls)
	spe = specificity(label,pred_cls)
	pre = precision_score(label, pred_cls)
	rec = recall_score(label, pred_cls)
	f1score=f1_score(label,pred_cls)
	rocauc = roc_auc_score(label, pred)
	prauc=average_precision_score(label, pred)
	mcc=matthews_corrcoef(label,pred_cls)
	epoch_loss = running_loss.get_average()
	running_loss.reset()
	return epoch_loss, acc, sen, spe, pre, rec, f1score, rocauc, prauc, mcc, label, pred

def plot_confusion_matrix(y_true, y_pred, save_dir, dataset_name='Validation'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-effective', 'Effective'],
                yticklabels=['Non-effective', 'Effective'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add metrics to the plot
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}',
             horizontalalignment='center', transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/confusion_matrix_{dataset_name.lower()}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_pr_curves(y_true, y_scores, save_dir, dataset_name='Validation'):
    """Plot ROC and PR curves side by side"""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {dataset_name}')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve - {dataset_name}')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_pr_curves_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/roc_pr_curves_{dataset_name.lower()}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(train_history, save_dir):
    """Plot training metrics over epochs"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Progress - Physics-Informed OligoFormer', fontsize=16)
    
    epochs = range(len(train_history['loss']))
    
    # Loss curves
    axes[0,0].plot(epochs, train_history['loss'], label='Train Loss', color='blue')
    axes[0,0].plot(epochs, train_history['val_loss'], label='Val Loss', color='red')
    axes[0,0].plot(epochs, train_history['physics_loss'], label='Physics Loss', color='green', linestyle='--')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Loss Curves')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[0,1].plot(epochs, train_history['val_rocauc'], label='Val ROC-AUC', color='red', marker='o', markersize=2)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('ROC-AUC')
    axes[0,1].set_title('ROC-AUC Progress')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim([0.65, 0.85])
    
    # F1 Score
    axes[0,2].plot(epochs, train_history['val_f1'], label='Val F1', color='purple', marker='s', markersize=2)
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('F1 Score')
    axes[0,2].set_title('F1 Score Progress')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_ylim([0.65, 0.80])
    
    # Precision-Recall
    axes[1,0].plot(epochs, train_history['val_precision'], label='Precision', color='orange')
    axes[1,0].plot(epochs, train_history['val_recall'], label='Recall', color='cyan')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Precision vs Recall')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # PR-AUC
    axes[1,1].plot(epochs, train_history['val_prauc'], label='Val PR-AUC', color='magenta', marker='^', markersize=2)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('PR-AUC')
    axes[1,1].set_title('PR-AUC Progress')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim([0.65, 0.85])
    
    # Accuracy
    axes[1,2].plot(epochs, train_history['val_accuracy'], label='Val Accuracy', color='brown', marker='d', markersize=2)
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('Accuracy')
    axes[1,2].set_title('Accuracy Progress')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim([0.60, 0.80])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/training_history.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_summary(metrics_dict, save_dir):
    """Create a bar plot summary of all metrics"""
    metrics = ['ROC-AUC', 'PR-AUC', 'F1 Score', 'Accuracy', 'Precision', 'Recall']
    values = [metrics_dict['rocauc'], metrics_dict['prauc'], metrics_dict['f1'],
              metrics_dict['accuracy'], metrics_dict['precision'], metrics_dict['recall']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylim([0, 1])
    plt.ylabel('Score')
    plt.title('Model Performance Summary')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/performance_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def train(Args):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
	random.seed(Args.seed)
	os.environ['PYTHONHASHSEED']=str(Args.seed)
	np.random.seed(Args.seed)
	if len(Args.datasets) >= 3:
    # Load and combine Hu + Taka for training
		train_df_hu = pd.read_csv(Args.path + Args.datasets[0] + '.csv', dtype=str)
		train_df_taka = pd.read_csv(Args.path + Args.datasets[2] + '.csv', dtype=str)
		train_df = pd.concat([train_df_hu, train_df_taka], ignore_index=True)
		# Use Mix for validation and test
		valid_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv', dtype=str)
		test_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv', dtype=str)
	else:
		train_df = pd.read_csv(Args.path + Args.datasets[0] + '.csv', dtype=str)
		valid_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv', dtype=str)
		test_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv', dtype=str)
	params = {'batch_size': Args.batch_size,
			'shuffle': True,
			'num_workers': 0,
			'drop_last': False}
	if not os.path.exists('./data/RNAFM'):
		os.system('bash scripts/RNA-FM-features.sh')
	if len(Args.datasets) >= 3:
    # Create combined dataloader - alternating between Hu and Taka
		train_ds_hu = DataLoader(data_process_loader(train_df_hu.index.values, train_df_hu.label.values,
							train_df_hu.y.values, train_df_hu, Args.datasets[0], Args.path), **params)
		train_ds_taka = DataLoader(data_process_loader(train_df_taka.index.values, train_df_taka.label.values,
								train_df_taka.y.values, train_df_taka, Args.datasets[2], Args.path), **params)
	else:
		train_ds = DataLoader(data_process_loader(train_df.index.values, train_df.label.values,
							train_df.y.values, train_df, Args.datasets[0], Args.path), **params)
	valid_ds = DataLoader(data_process_loader(valid_df.index.values, valid_df.label.values,valid_df.y.values, valid_df, Args.datasets[1],Args.path),**params)
	test_ds = DataLoader(data_process_loader(test_df.index.values, test_df.label.values,test_df.y.values, test_df, Args.datasets[1],Args.path), **params)
	OFmodel = Oligo(vocab_size = Args.vocab_size, embedding_dim = Args.embedding_dim, lstm_dim = Args.lstm_dim,  n_head = Args.n_head, n_layers = Args.n_layers, lm1 = Args.lm1, lm2 = Args.lm2).to(device)
	if Args.resume is not None:
		OFmodel.load_state_dict(torch.load(Args.resume,map_location=device))
	criterion = nn.MSELoss()
	best_AUC = 0.0
	best_loss = 1e10
	best_epoch = 0
	tolerence_epoch = Args.early_stopping
	# Replace line 207 with differential learning rates
	optimizer = optim.Adam([
    {'params': OFmodel.siRNA_encoder.position_weights, 'lr': Args.learning_rate * 5},  
    {'params': [p for n, p in OFmodel.named_parameters() if 'position_weights' not in n], 'lr': Args.learning_rate}
], lr=Args.learning_rate)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = Args.weight_decay)
	params = dict(
		data_path=Args.path,
		save_dir=Args.output_dir,
		dataset=Args.datasets[1],
		batch_size=Args.batch_size
	)
	logger = TrainLogger(params)
	logger.info("pyTorch!!!")
	if len(Args.datasets) >= 3:
		logger.info(f"Number of train: {train_df.shape[0]} (Hu: {train_df_hu.shape[0]}, Taka: {train_df_taka.shape[0]})")
	else:
		logger.info(f"Number of train: {train_df.shape[0]}")
	logger.info(f"Number of val: {valid_df.shape[0]}")
	logger.info(f"Number of test: {test_df.shape[0]}")
	print('-----------------Start training!-----------------')
	running_loss = AverageMeter()
	physics_loss_tracker = AverageMeter()  # ADD THIS 
      
	# Initialize history tracking
	train_history = {
			'loss': [], 'val_loss': [], 'physics_loss': [],
			'val_rocauc': [], 'val_prauc': [], 'val_f1': [],
			'val_accuracy': [], 'val_precision': [], 'val_recall': []
					}
      
	# Variables to store best model predictions
	best_val_label = None
	best_val_pred = None
	best_val_pred_cls = None
	for epoch in range(Args.epoch):
		physics_loss_tracker.reset()
		
		if len(Args.datasets) >= 3:
			# Combine both datasets in each epoch
			max_batches = max(len(train_ds_hu), len(train_ds_taka))
			train_ds_hu_cycle = cycle(train_ds_hu)
			train_ds_taka_cycle = cycle(train_ds_taka)
			
			for _ in range(max_batches):
				# Train on batch from Hu
				data_hu = next(train_ds_hu_cycle)
				siRNA = data_hu[0].to(device)
				mRNA = data_hu[1].to(device)
				siRNA_FM = data_hu[2].to(device)
				mRNA_FM = data_hu[3].to(device)
				label = data_hu[4].to(device)
				td = data_hu[6].to(device)
				siRNA.requires_grad = True
				output, siRNA_attention, mRNA_attention = OFmodel(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
				ce_loss = criterion(output[:,1],label.float())
				physics_loss = calculate_physics_loss(siRNA, epoch)
				loss = ce_loss + physics_loss
				optimizer.zero_grad()
				loss.backward()
				running_loss.update(loss, output.shape[0])
				physics_loss_tracker.update(physics_loss, output.shape[0])
				optimizer.step()
				
				# Train on batch from Taka
				data_taka = next(train_ds_taka_cycle)
				siRNA = data_taka[0].to(device)
				mRNA = data_taka[1].to(device)
				siRNA_FM = data_taka[2].to(device)
				mRNA_FM = data_taka[3].to(device)
				label = data_taka[4].to(device)
				td = data_taka[6].to(device)
				siRNA.requires_grad = True
				output, siRNA_attention, mRNA_attention = OFmodel(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
				ce_loss = criterion(output[:,1],label.float())
				physics_loss = calculate_physics_loss(siRNA, epoch)
				loss = ce_loss + physics_loss
				optimizer.zero_grad()
				loss.backward()
				running_loss.update(loss, output.shape[0])
				physics_loss_tracker.update(physics_loss, output.shape[0])
				optimizer.step()
		else:
			# Original single dataset training
			for i, data in enumerate(train_ds):
				siRNA = data[0].to(device)
				mRNA = data[1].to(device)
				siRNA_FM = data[2].to(device)
				mRNA_FM = data[3].to(device)
				label = data[4].to(device)
				td = data[6].to(device)
				siRNA.requires_grad = True
				output, siRNA_attention, mRNA_attention = OFmodel(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
				ce_loss = criterion(output[:,1],label.float())
				physics_loss = calculate_physics_loss(siRNA, epoch)
				loss = ce_loss + physics_loss
				optimizer.zero_grad()
				loss.backward()
				running_loss.update(loss, output.shape[0])
				physics_loss_tracker.update(physics_loss, output.shape[0])
				optimizer.step()
		scheduler.step()
		torch.cuda.empty_cache()
		epoch_loss = running_loss.get_average()
		running_loss.reset()
		#train_loss, train_acc, train_sen, train_spe, train_pre, train_rec, train_f1, train_rocauc, train_prauc, train_mcc, train_label, train_pred = val(OFmodel, criterion, train_ds)
		val_loss, val_acc, val_sen, val_spe, val_pre, val_rec, val_f1, val_rocauc, val_prauc, val_mcc, val_label, val_pred = val(OFmodel, criterion, valid_ds)
		test_loss, test_acc, test_sen, test_spe, test_pre, test_rec, test_f1, test_rocauc, test_prauc, test_mcc, test_label, test_pred = val(OFmodel, criterion, test_ds)
        # Store metrics
		train_history['loss'].append(float(epoch_loss))
		train_history['val_loss'].append(float(val_loss))
		avg_physics = physics_loss_tracker.get_average()
		if isinstance(avg_physics, torch.Tensor):
			avg_physics = avg_physics.cpu().numpy()
		train_history['physics_loss'].append(float(avg_physics))
		train_history['val_rocauc'].append(val_rocauc)
		train_history['val_prauc'].append(val_prauc)
		train_history['val_f1'].append(val_f1)
		train_history['val_accuracy'].append(val_acc)
		train_history['val_precision'].append(val_pre)
		train_history['val_recall'].append(val_rec)
		if  val_loss < best_loss and val_rocauc > best_AUC: 
			best_AUC = val_rocauc
			best_loss = val_loss
			best_epoch = epoch
                  
            # Store best predictions
			best_val_label = val_label
			best_val_pred = val_pred
			best_val_pred_cls = (val_pred > 0.5).astype(int)
                  
			avg_physics = physics_loss_tracker.get_average() 
			msg = "epoch-%d, loss-%.4f, physics-%.4f, val_loss-%.4f, val_acc-%.4f,val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_f1-%.4f ***" % (epoch, epoch_loss, avg_physics, val_loss, val_acc, val_pre, val_rec,val_rocauc,val_prauc,val_f1)			
			torch.save(OFmodel.state_dict(), os.path.join(logger.get_model_dir(), msg+'.pth'))
		else:
			avg_physics = physics_loss_tracker.get_average()  # ADDED THIS
			msg = "epoch-%d, loss-%.4f, physics-%.4f, val_loss-%.4f, val_acc-%.4f,val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_f1-%.4f" % (epoch, epoch_loss, avg_physics, val_loss, val_acc, val_pre, val_rec,val_rocauc,val_prauc,val_f1)
		logger.info(msg)
            
		# Plot every 20 epochs
		if (epoch + 1) % 20 == 0:
			plot_training_history(train_history, logger.get_model_dir())
		
		if epoch - best_epoch > tolerence_epoch:
			break
	# After training completes, generate all final plots
	print("Generating final plots...")
	model_dir = logger.get_model_dir()

	# Plot training history
	plot_training_history(train_history, model_dir)

	# Plot confusion matrix for best model
	plot_confusion_matrix(best_val_label, best_val_pred_cls, model_dir, 'Best_Validation')

	# Plot ROC and PR curves
	plot_roc_pr_curves(best_val_label, best_val_pred, model_dir, 'Best_Validation')

	# Plot performance summary
	best_metrics = {
		'accuracy': accuracy_score(best_val_label, best_val_pred_cls),
		'precision': precision_score(best_val_label, best_val_pred_cls),
		'recall': recall_score(best_val_label, best_val_pred_cls),
		'f1': f1_score(best_val_label, best_val_pred_cls),
		'rocauc': roc_auc_score(best_val_label, best_val_pred),
		'prauc': average_precision_score(best_val_label, best_val_pred)
	}
	plot_performance_summary(best_metrics, model_dir)

	# Save metrics to CSV
	metrics_df = pd.DataFrame(train_history)
	metrics_df.to_csv(f'{model_dir}/training_metrics.csv', index=False)
	print(f"All plots saved to {model_dir}")