import torch
import torch.nn as nn
import torch.nn.functional as F

class DKTPlusModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device,
                 lambda_w1=0.0, lambda_w2=0.0, lambda_o=0.0):
        super(DKTPlusModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.device = device

        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
        self.lambda_o = lambda_o

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size, device=self.device)
        out, _ = self.rnn(x, h0)  # [batch, seq_len, hidden_dim]
        preds = self.sigmoid(self.fc(out))  # [batch, seq_len, num_questions]
        return preds

    def compute_loss(self, preds, target_labels, target_mask, x_corr=None):
        loss_main = F.binary_cross_entropy(preds[target_mask], target_labels[target_mask])

        total_loss = loss_main

        if self.lambda_o > 0.0 and x_corr is not None:
            recon_loss = F.binary_cross_entropy(preds[target_mask], x_corr[target_mask])
            total_loss += self.lambda_o * recon_loss


        if self.lambda_w1 > 0.0:
            waviness_l1 = torch.abs(preds[:, 1:, :] - preds[:, :-1, :])
            total_w1 = waviness_l1.sum() / preds.size(0) / preds.size(1)
            total_loss += self.lambda_w1 * total_w1

        if self.lambda_w2 > 0.0:
            waviness_l2 = (preds[:, 1:, :] - preds[:, :-1, :]) ** 2
            total_w2 = waviness_l2.sum() / preds.size(0) / preds.size(1)
            total_loss += self.lambda_w2 * total_w2

        return total_loss
