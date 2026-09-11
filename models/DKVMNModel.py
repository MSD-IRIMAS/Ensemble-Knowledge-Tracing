import torch
import torch.nn as nn
import torch.nn.functional as F

class DKVMNModel(nn.Module):
    def __init__(self, input_dim, memory_size, memory_dim, output_dim, device):
        super(DKVMNModel, self).__init__()
        self.device = device
        self.memory_size = memory_size  # nombre de "concepts"
        self.memory_dim = memory_dim    # dimension de chaque slot de mémoire
        self.output_dim = output_dim

        # Mémoire clé (fixe)
        self.memory_key = nn.Parameter(torch.Tensor(memory_size, memory_dim))
        nn.init.xavier_uniform_(self.memory_key)

        # Mémoire valeur (mise à jour au fil du temps)
        self.memory_value = nn.Parameter(torch.Tensor(memory_size, memory_dim))
        nn.init.xavier_uniform_(self.memory_value)

        # Mécanismes d'attention et de lecture/écriture
        self.read_linear = nn.Linear(input_dim, memory_dim)
        self.write_linear = nn.Linear(input_dim, memory_dim)
        self.predict_layer = nn.Linear(memory_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        predictions = []

        memory_value = self.memory_value.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, memory_size, memory_dim]

        for t in range(seq_len):
            xt = x[:, t, :]  # [batch_size, input_dim]
            k_t = self.read_linear(xt)  # [batch_size, memory_dim]
            w_t = F.softmax(torch.matmul(k_t.unsqueeze(1), self.memory_key.T), dim=-1)  # [batch_size, 1, memory_size]

            # Lecture : weighted sum des slots de mémoire
            r_t = torch.matmul(w_t, memory_value).squeeze(1)  # [batch_size, memory_dim]
            y_t = self.sigmoid(self.predict_layer(r_t))  # [batch_size, output_dim]
            predictions.append(y_t.unsqueeze(1))

            # Écriture : update mémoire valeur
            erase = torch.sigmoid(self.write_linear(xt)).unsqueeze(1)  # [batch_size, 1, memory_dim]
            add = torch.tanh(self.write_linear(xt)).unsqueeze(1)
            w_t_trans = w_t.transpose(1, 2)  # [batch_size, memory_size, 1]

            memory_value = memory_value * (1 - w_t_trans * erase) + w_t_trans * add

        return torch.cat(predictions, dim=1)  # [batch_size, seq_len, output_dim]
