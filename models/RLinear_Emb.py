import torch
import torch.nn as nn
from torch.nn import functional as F

from layers.HeEmb import HeEmb
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.emb = HeEmb(
            n = configs.enc_in, 
            input_dim=configs.seq_len, 
            output_dim = configs.pred_len,
            model_type=configs.model_type,
            rank=configs.rank,
            num_experts=configs.num_experts,
            moe_router_type=configs.moe_router_type,
            moe_mlp_hidden_dim=configs.moe_mlp_hidden_dim,
            channel_identity_dim=configs.channel_identity_dim
            )


    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - mean) / std

        x = x.permute(0,2,1)
        pred = self.emb(x)

        pred = pred.permute(0,2,1)
        
        pred = pred * std + mean

        return pred
        
