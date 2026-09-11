import torch
import torch.nn as nn

class FCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(FCNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.device = device

        layers = []
        in_dim = input_dim
        for _ in range(layer_dim):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.fcn = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        x = x.view(-1, input_dim)  # aplatissement pour traiter chaque timestep indépendamment
        out = self.fcn(x)  # [batch_size * seq_len, hidden_dim]
        out = self.fc_out(out)
        out = self.sigmoid(out)  # [batch_size * seq_len, output_dim]
        return out.view(batch_size, seq_len, self.output_dim)
