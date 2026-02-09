import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import OrderedDict
import math
from physics_cache import set_nt_probs_cache

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    def forward(self, x):
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))
    def forward(self, x):
        return self.inc(x).squeeze(-1)

class siRNA_Encoder2(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx+1, embedding_dim, 96, 3)
            )
        self.linear = nn.Linear(block_num * 96, 96)
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # [batch_size = 128, seq_len = 30]
        return self.encoding[:seq_len, :]

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
    def forward(self, x):
        # tok_emb = self.tok_emb(x)
        tok_emb = x
        pos_emb = self.pos_emb(x)
        return tok_emb + pos_emb

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.12)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out,attention
    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor
    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout = nn.Dropout(0.12)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.norm2 = LayerNorm(d_model=d_model)
    def forward(self, embed, src_mask):
        attn_out, attention = self.attention(q=embed, k=embed, v=embed, mask=src_mask)  # attention: [B,H,L,L]
        x = self.norm1(embed + self.dropout(attn_out))  # residual + norm
        ff = self.ffn(x)
        x = self.norm2(x + ff)                          # residual + norm
        return x, attention

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,max_len=max_len,vocab_size=enc_voc_size)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,n_head=n_head) for _ in range(n_layers)])
    def forward(self, x, src_mask):
        x = self.emb(x)              # [B, L, D]
        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x, src_mask)
        return x, last_attn           # attention shape: [B, H, L, L]

class siRNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim, n_head, n_layers):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 64, kernel_size=(1,13), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.12)
        self.avgpool = nn.AvgPool1d(kernel_size = 2)
        self.maxpool = nn.MaxPool1d(kernel_size = 2)
        self.biLSTM = nn.LSTM(input_size=77, hidden_size=lstm_dim, num_layers=2,
                      bidirectional=True, batch_first=True) # 68
        self.encoder = Encoder(vocab_size,19,lstm_dim*2,embedding_dim,n_head,n_layers)
        self.position_weights = nn.Parameter(torch.ones(19, dtype=torch.float32).to(device))

        self.nt_head = nn.Sequential(
            nn.Linear(77, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.float() # 16,1,19,5
        x0 = x.squeeze(1) # 16,19,5
        x = self.conv2d(x) # 16,64,19,1
        x = self.relu(x) # 16,64,19,1
        x = x.squeeze(-1) # 16,64,19
        x = x.transpose(1,2) # 16,19,64
        x1 = self.avgpool(x) # 16,19,32
        x2 = self.maxpool(x) # 16,19,32
        x = torch.cat([x0,x1,x2],dim = -1) # 16,19,69
        nt_logits = self.nt_head(x)           # [B, 19, 4]
        nt_probs  = F.softmax(nt_logits, dim=-1)
        set_nt_probs_cache(nt_probs) 
        #_set_nt_probs_cache(nt_probs)
        x, _ = self.biLSTM(x) # 16,19,64
        x = self.relu(x) # 16,19,64
        x = self.dropout(x) # 16,19,64 #######
        #x = x * self.position_weights.unsqueeze(0).unsqueeze(-1)
        x, attention = self.encoder(x,src_mask = None) # 16,19,64
        return x, attention, nt_probs
    
            

class mRNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim, n_head, n_layers, lm1,lm2):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 64, kernel_size=(5,5), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.12)
        self.avgpool = nn.AvgPool1d(kernel_size = 2) #2
        self.maxpool = nn.MaxPool1d(kernel_size = 2) #2
        self.biLSTM = nn.LSTM(input_size=64, hidden_size=lstm_dim, num_layers=2,
                      bidirectional=True, batch_first=True)# 68
        self.encoder = Encoder(vocab_size,19 + lm1 + lm2,lstm_dim*2,embedding_dim,n_head,n_layers)

    def forward(self, x):
        x = x.float() # 16,1,57,5
        x0 = x.squeeze(1) # 16,57,5
        x = self.conv2d(x) # 16,64,57,1
        x = self.relu(x) # 16,64,57,1
        x = x.squeeze(-1) # 16,64,57
        x = x.transpose(1,2) # 16,57,64
        x1 = self.avgpool(x) # 16,57,32
        x2 = self.maxpool(x) # 16,57,32
        x = torch.cat([x1,x2],dim = -1) # 16,57,64 x0,
        x, _ = self.biLSTM(x) # 16,57,64
        x = self.relu(x) # 16,57,64
        x = self.dropout(x) # 16,57,64 #######
        x, attention = self.encoder(x,src_mask = None) # 16,53,64
        return x, attention

class Oligo(nn.Module):
    def __init__(self, vocab_size = 26, embedding_dim = 128, lstm_dim = 32,  n_head = 8, n_layers = 1, lm1 = 19, lm2 = 19, td_dim=35): # 4928
        super().__init__()
        self.siRNA_encoder = siRNA_Encoder(vocab_size, embedding_dim, lstm_dim, n_head, n_layers)
        self.mRNA_encoder = mRNA_Encoder(vocab_size, embedding_dim, lstm_dim, n_head, n_layers, lm1, lm2)
        self.si_to_mr = MultiHeadAttention(d_model=2*lstm_dim, n_head=n_head)  # Q=siRNA, K/V=mRNA
        self.mr_to_si = MultiHeadAttention(d_model=2*lstm_dim, n_head=n_head)  # Q=mRNA, K/V=siRNA
        self.seq_pool = lambda x: torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)

        self.siRNA_avgpool = nn.AvgPool2d((19, 5))
        self.mRNA_avgpool = nn.AvgPool2d((19 + lm1 + lm2, 5))

        # Encoders → cross-attn → mean+max pooling gives 4*lstm_dim per stream
        enc_dim = 8 * lstm_dim  # si_vec (4*lstm_dim) + mr_vec (4*lstm_dim)
        # Infer FM pooled dims once using dummy tensors
        with torch.no_grad():
            B = 2
            si_fm_pooled = self.siRNA_avgpool(torch.zeros(B, 1, 19, 5))
            mr_fm_pooled = self.mRNA_avgpool(torch.zeros(B, 1, 19 + lm1 + lm2, 5))
            siRNA_FM_dim = si_fm_pooled.view(B, -1).shape[1]   # usually 1 if C=1
            mRNA_FM_dim  = mr_fm_pooled.view(B, -1).shape[1]   # usually 1 if C=1

        # If your thermo vector length differs, pass it via td_dim arg
        in_dim = 16 * lstm_dim + td_dim + siRNA_FM_dim + mRNA_FM_dim  # 16*32=512
        # ---------------------------------------
        self.classifier = nn.Sequential(
            nn.LazyLinear(256),           
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 2)
            #nn.Softmax()
        )
        # Auxiliary heads for multi-task learning
        self.gc_head = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.seed_au_head = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.asymmetry_head = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Asymmetry can be negative
        )
        
        self.flatten = nn.Flatten()
    def forward(self, siRNA, mRNA, siRNA_FM, mRNA_FM,td):
        siRNA, siRNA_attention, si_nt_probs = self.siRNA_encoder(siRNA)
        mRNA, mRNA_attention = self.mRNA_encoder(mRNA)
        siRNA_cross, si_to_mr_map = self.si_to_mr(q=siRNA, k=mRNA,  v=mRNA)   # [B, Ls, 2*lstm_dim], [B,H,Ls,Lm]
        mRNA_cross,  mr_to_si_map = self.mr_to_si(q=mRNA,  k=siRNA, v=siRNA)  # [B, Lm, 2*lstm_dim], [B,H,Lm,Ls]
        si_vec = self.seq_pool(siRNA_cross)   # [B, 4*lstm_dim] (mean+max)
        mr_vec = self.seq_pool(mRNA_cross)    # [B, 4*lstm_dim]
        siRNA_FM = self.siRNA_avgpool(siRNA_FM)
        siRNA_FM = siRNA_FM.view(siRNA_FM.shape[0], siRNA_FM.shape[2])
        mRNA_FM = self.mRNA_avgpool(mRNA_FM)
        mRNA_FM = mRNA_FM.view(mRNA_FM.shape[0], mRNA_FM.shape[2])
        siRNA = self.flatten(siRNA)
        mRNA = self.flatten(mRNA)
        siRNA_FM = self.flatten(siRNA_FM)
        mRNA_FM = self.flatten(mRNA_FM)
        td = self.flatten(td)
        merge = torch.cat([si_vec, mr_vec, siRNA_FM, mRNA_FM, td], dim=-1)
        x = self.classifier(merge)
        
        # Auxiliary predictions for multi-task learning
        gc_pred = self.gc_head(merge).squeeze(-1)
        seed_au_pred = self.seed_au_head(merge).squeeze(-1)
        asymmetry_pred = self.asymmetry_head(merge).squeeze(-1)
        aux_preds = (gc_pred, seed_au_pred, asymmetry_pred)
        
        return x, si_to_mr_map, mr_to_si_map, si_nt_probs, aux_preds