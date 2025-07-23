import torch
import torch.nn as nn
from torch.nn import functional as F
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.emb=nn.Linear(configs.seq_len, configs.d_model)

        self.temporal = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x):
        
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - mean) / std
        x = x.permute(0,2,1)

        x = self.emb(x)
        x = self.temporal(x) + x
        pred = self.projection(x)

        pred = pred.permute(0,2,1)
        
        pred = pred * std + mean

        return pred
        
