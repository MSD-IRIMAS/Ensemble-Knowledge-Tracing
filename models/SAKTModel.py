import math
import torch
import copy
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, h, length, d_model, dropout):
        super(Encoder, self).__init__()
        self.multi_headed_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_model * 4)
        self.sublayer = clones(SublayerConnection(length, d_model, dropout), 2)

    def forward(self, x, y, mask=None):
        attn_output, _ = self.multi_headed_attention(y, x, x, attn_mask=mask)
        y = self.sublayer[0](y, lambda y: attn_output)
        return self.sublayer[1](y, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, length, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=[length, d_model])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embedding(nn.Module):
    def __init__(self, n_questions, length, embedding_dim, input_dim=None):
        super().__init__()
        self.n_questions = n_questions
        self.compressed = input_dim is not None and input_dim != n_questions * 2
        if self.compressed:
            self.x_emb = nn.Linear(input_dim, embedding_dim, bias=False)
            self.y_emb = nn.Linear(input_dim, embedding_dim, bias=False)
        else:
            self.x_emb = nn.Linear(n_questions, embedding_dim, bias=False)
            self.y_emb = nn.Linear(n_questions * 2, embedding_dim, bias=False)
        self.pos_emb = nn.Embedding(length, embedding_dim)
        self.length = length

    def forward(self, y):  # shape: [batch_size, length, input_dim]
        n_batch = y.shape[0]
        device = y.device
        if self.compressed:
            x = y
        else:
            x = y[:, :, 0:self.n_questions] + y[:, :, self.n_questions:]
        p = torch.arange(self.length, device=device).repeat(n_batch, 1)
        pos = self.pos_emb(p)
        y = self.y_emb(y)  # [batch_size, length, embedding_dim]
        x = self.x_emb(x)  # [batch_size, length, embedding_dim]
        return (x + pos, y)

class SAKTModel(nn.Module):
    def __init__(self, h, length, d_model, n_question, dropout, input_dim=None):
        super(SAKTModel, self).__init__()
        self.embedding = Embedding(n_question, length, d_model, input_dim=input_dim)
        self.encoder = Encoder(h, length, d_model, dropout)
        self.w = nn.Linear(d_model, n_question)
        self.sig = nn.Sigmoid()

    def forward(self, y):  # shape of input: [batch_size, length, questions * 2]
        x, y = self.embedding(y)  # shape: [batch_size, length, d_model]
        encode = self.encoder(x, y)  # shape: [batch_size, length, d_model]
        res = self.sig(self.w(encode))
        return res
