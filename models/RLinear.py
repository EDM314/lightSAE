import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.Linear(configs.seq_len, configs.pred_len)


    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True) + 1e-6
        x = (x - mean) / std

        pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        
        pred = pred * std + mean

        return pred
