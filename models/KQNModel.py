import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc

class KQN(nn.Module):
  # n_skills: number of skills in dataset
  # n_hidden: dimensionality of skill and knowledge state vectors
  # n_rnn_hidden: number of hidden units in rnn knowledge encoder
  # n_mlp_hidden: number of hidden units in mlp skill encoder
  # n_rnn_layers: number of layers in rnn knowledge encoder
  # rnn_type: type of rnn cell, chosen from ['gru', 'lstm']

  def __init__(self, n_skills:int, n_hidden:int, n_rnn_hidden:int, n_mlp_hidden:int, n_rnn_layers:int=1, rnn_type='lstm', device='cpu', rnn_input_dim:int=None):
    super(KQN, self).__init__()
    self.device = device
    self.n_hidden = n_hidden
    self.n_rnn_hidden = n_rnn_hidden
    self.n_mlp_hidden = n_mlp_hidden
    self.n_rnn_layers = n_rnn_layers
    
    self.rnn_type, rnn_type = rnn_type.lower(), rnn_type.lower()
    rnn_input_size = rnn_input_dim if rnn_input_dim is not None else 2 * n_skills

    if rnn_type == 'lstm':
      self.rnn = nn.LSTM(
        input_size=rnn_input_size,
        hidden_size=n_rnn_hidden,
        num_layers=n_rnn_layers,
        batch_first=True
      )
    elif rnn_type == 'gru':
      self.rnn = nn.GRU(
        input_size=rnn_input_size,
        hidden_size=n_rnn_hidden,
        num_layers=n_rnn_layers,
        batch_first=True
      )
      
    self.linear = nn.Linear(n_rnn_hidden, n_hidden)
    
    # Embedding lookup is equivalent to Linear(one_hot) but uses O(1) memory per token
    # instead of O(n_skills) — critical for large datasets (KT2: 18144 skills)
    self.skill_emb  = nn.Embedding(n_skills, n_mlp_hidden)
    self.skill_proj = nn.Sequential(
      nn.ReLU(),
      nn.Linear(n_mlp_hidden, n_hidden),
      nn.ReLU()
    )
    
    self.sigmoid = nn.Sigmoid()
    self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
  def init_hidden(self, batch_size: int):
    weight = next(self.parameters()).data
    if self.rnn_type == 'lstm':
        return (Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()),
                Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()))
    else:
        return Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_())
    
  def encode_knowledge(self, in_data:torch.FloatTensor, seq_len:torch.LongTensor):
    batch_size = in_data.size(0)
    self.hidden = self.init_hidden(batch_size)
    
    rnn_input = pack_padded_sequence(in_data, seq_len.cpu(), batch_first=True, enforce_sorted=False)


    rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
    rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True) # (batch_size, max_seq_len, n_rnn_hidden)
    encoded_knowledge = self.linear(rnn_output) # (batch_size, max_seq_len, n_hidden)
    return encoded_knowledge
  
  def encode_skills(self, next_skills: torch.LongTensor):
    # next_skills: (batch, seq) integer indices of next question
    embedded       = self.skill_emb(next_skills)   # (batch, seq, n_mlp_hidden)
    encoded_skills = self.skill_proj(embedded)      # (batch, seq, n_hidden)
    encoded_skills = F.normalize(encoded_skills, p=2, dim=2)
    return encoded_skills
  
  def selecting_mask(self, seq_len:torch.LongTensor, max_seq_len:int):
    # given seq_len tensor of size (batch_size,) get gathering indices
    # i.e., for each sample in the batch, gather results up to its sequence length out of max_seq_len results
    batch_size = seq_len.size(0)
    mask = torch.arange(max_seq_len, device=self.device).repeat(batch_size, 1) < seq_len.unsqueeze(1)
    
    return mask
    
  def forward(self, in_data: torch.FloatTensor, seq_len: torch.LongTensor, next_skills: torch.FloatTensor):
    encoded_knowledge = self.encode_knowledge(in_data, seq_len)  # (batch_size, ?, n_hidden)
    encoded_skills = self.encode_skills(next_skills)             # (batch_size, ?, n_hidden)

    # S'assurer que les deux séquences ont la même longueur
    min_len = min(encoded_knowledge.size(1), encoded_skills.size(1))
    encoded_knowledge = encoded_knowledge[:, :min_len, :]
    encoded_skills = encoded_skills[:, :min_len, :]

    logits = torch.sum(encoded_knowledge * encoded_skills, dim=2)  # (batch_size, min_len)
    return logits

  
  def loss(self, logits: torch.FloatTensor, correctness: torch.FloatTensor, mask: torch.Tensor):
    # Tronquer mask et correctness pour correspondre à logits
    min_len = logits.size(1)
    mask = mask[:, :min_len]
    correctness = correctness[:, :min_len]

    logits = logits.masked_select(mask)
    correctness = correctness.masked_select(mask)
    bce_loss = self.loss_fn(logits, correctness)
    return bce_loss

  
  def calculate_auc(self, preds, labels, pos_label=1):
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=pos_label)
    auc_val = auc(fpr, tpr)
    return auc_val
  
  def auc(self, logits, correctness):
    preds = self.sigmoid(logits)
    auc_val = self.calculate_auc(preds, correctness)
    return auc_val