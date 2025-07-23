import torch
import torch.nn as nn
from layers.HeEmb import HeEmb


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = HeEmb(configs.enc_in, configs.seq_len, configs.pred_len)


    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True) + 1e-6
        x = (x - mean) / std

        x = x.permute(0, 2, 1)
        pred = self.Linear(x)
        pred = pred.permute(0, 2, 1)
       
        pred = pred * std + mean

        return pred
