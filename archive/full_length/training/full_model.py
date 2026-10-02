'''Mimosa without the 40-nt limit.

Differences from the original Transformer in Mimosa.py:
  - the final Linear(40, 12) is replaced by masked pooling over mRNA positions, so any
    input length works ('mean' stays closest to the original, which averaged before the
    linear layer; 'max' matches Mimosa's "at least one site" rule)
  - the learned position encoding covers max_len instead of 100 positions
  - padding masks are passed to the encoder, the cross-attention and the pooling
  - the forward returns logits; the original applied softmax before CrossEntropyLoss,
    which applies it again
'''
import torch
import torch.nn as nn


class FullLengthMimosa(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=16, num_heads=8, dropout=0.1,
                 output_size=2, max_len=4096, pooling='mean'):
        super().__init__()
        self.pooling = pooling
        self.embedding_m = nn.Embedding(input_size, hidden_size)
        self.position_encoding_m = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        self.interaction_embedding_m = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_m, mean=0, std=0.1)

        self.embedding_mi = nn.Embedding(input_size, hidden_size)
        self.position_encoding_mi = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_mi = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_mi, mean=0, std=0.1)

        encoder_layers_m = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_m = nn.TransformerEncoder(encoder_layers_m, num_layers)
        encoder_layers_mi = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_mi = nn.TransformerEncoder(encoder_layers_mi, num_layers)

        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc1 = nn.Linear(hidden_size, 12)
        self.fc2 = nn.Linear(12, output_size)

    def forward(self, emb_m, emb_mi, pairing_m, pairing_mi, pad_mask=None):
        m = (self.embedding_m(emb_m) + self.position_encoding_m[:, :emb_m.size(1), :]
             + self.interaction_embedding_m(pairing_m))
        mi = (self.embedding_mi(emb_mi) + self.position_encoding_mi[:, :emb_mi.size(1), :]
              + self.interaction_embedding_mi(pairing_mi))
        m = self.encoder_m(m.permute(1, 0, 2), src_key_padding_mask=pad_mask)
        mi = self.encoder_mi(mi.permute(1, 0, 2))
        cross, _ = self.cross_attention(m, mi, mi)          # (L, B, H), query = mRNA
        cross = cross.permute(1, 0, 2)                      # (B, L, H)
        if self.pooling == 'max':
            if pad_mask is not None:
                cross = cross.masked_fill(pad_mask.unsqueeze(-1), float('-inf'))
            pooled = cross.max(dim=1).values
        else:
            if pad_mask is not None:
                keep = (~pad_mask).unsqueeze(-1).to(cross.dtype)
                pooled = (cross * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
            else:
                pooled = cross.mean(dim=1)
        return self.fc2(torch.relu(self.fc1(pooled)))
