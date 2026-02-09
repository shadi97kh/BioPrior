from train_single import train_single
from test_single import test_single
from test import test
from scripts.train import train_intra_5fold
from infer import infer
from mismatch import mismatch
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='OligoFormer')
    # Data options
    parser.add_argument('--datasets', type=str, nargs='+', default=['Hu','Mix'], 
                        help="Datasets: ['Hu','Mix','Taka','Shabalina']")
    parser.add_argument('--val_mode', type=str, default='inter', choices=['intra', 'inter', 'lodo'], 
                        help='intra (5-fold CV), inter (train/test split), lodo (leave-one-out)')
    parser.add_argument('--path', type=str, default='./data/', help='data path')
    parser.add_argument('--output_dir', type=str, default="result/", help='output directory')
    parser.add_argument('--best_model', type=str, default="./model/best_model.pth")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=0.999)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=26)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--lstm_dim', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lm', type=int, default=19)
    parser.add_argument('--lm1', type=int, default=19)
    parser.add_argument('--lm2', type=int, default=19)

    # Options
    parser.add_argument('-t','--test', action='store_true')
    parser.add_argument('-s','--single', action='store_true')
    parser.add_argument('--physics_weight', type=float, default=0.05)
    parser.add_argument('--data_fraction', type=float, default=1.0)

    # Infer module
    parser.add_argument('-i','--infer', type=int, default=0)
    parser.add_argument('-top','--top_n', type=int, default=-1)
    parser.add_argument('-i1','--infer_fasta', type=str, default='./data/example.fa')
    parser.add_argument('-i2','--infer_siRNA_fasta', nargs='?', const=False)
    parser.add_argument('-e', '--eval', type=str, nargs='+', default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('-nf', '--no_func', action='store_true')

    # Ablation mode - simplified
    parser.add_argument('--ablation', type=str, default='mechanistic',
                        choices=['baseline', 'mechanistic', 'full', 'neutral'],
                        help='baseline=no physics, mechanistic=full physics')
    
    # Off-target module
    parser.add_argument('-off','--off_target', action='store_true')
    parser.add_argument('-a','--all_human', action='store_true')
    parser.add_argument('--utr', type=str, default='./off-target/ref/100_human_UTR.txt')
    parser.add_argument('--orf', type=str, default='./off-target/ref/100_human_ORF.txt')
    parser.add_argument('--pita_threshold', type=float, default=-10)
    parser.add_argument('--targetscan_threshold', type=float, default=1)
    
    # Toxicity
    parser.add_argument('-tox','--toxicity', action='store_true')
    parser.add_argument('--toxicity_threshold', type=float, default=50.0)
    
    # Mismatch
    parser.add_argument('-m','--mismatch', type=int, default=0)

    Args = parser.parse_args()
    
    if not os.path.exists('./result'):
        os.mkdir('./result')

    if Args.mismatch > 0:
        mismatch(Args)
    elif Args.infer > 0:
        infer(Args)
    elif Args.eval:
        from infer import evaluate_datasets
        evaluate_datasets(Args)
    elif Args.test:
        test_single(Args) if Args.single else test(Args)
    else:
        if Args.val_mode == 'intra':
            train_intra_5fold(Args)
        elif Args.val_mode == 'lodo':
            from scripts.train import train_lodo
            train_lodo(Args)
        else:
            from scripts.train import train
            train_single(Args) if Args.single else train(Args)
        

if __name__ == '__main__':
    main()