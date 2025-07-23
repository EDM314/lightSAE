import torch
import torch.nn as nn
from torch.nn import functional as F

from layers.HeEmb import HeEmb
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_revin = configs.revin
        self.inout_type = configs.inout_type

        if self.inout_type == 'in':
            self.emb = HeEmb(
                n = configs.enc_in, 
                input_dim=configs.seq_len, 
                output_dim = configs.d_model, 
                model_type = configs.model_type,
                rank=configs.rank,
                num_experts=configs.num_experts,
                moe_router_type=configs.moe_router_type,
                moe_mlp_hidden_dim=configs.moe_mlp_hidden_dim,
                channel_identity_dim=configs.channel_identity_dim,
                use_softmax=configs.use_softmax,
                grouped_bias=configs.grouped_bias
                )
        elif self.inout_type == 'out':
            self.emb = nn.Linear(configs.seq_len, configs.d_model, bias=True)
        elif self.inout_type == 'inout':
            self.emb = HeEmb(
                n = configs.enc_in, 
                input_dim=configs.seq_len, 
                output_dim = configs.d_model, 
                model_type = configs.model_type,
                rank=configs.rank,
                num_experts=configs.num_experts,
                moe_router_type=configs.moe_router_type,
                moe_mlp_hidden_dim=configs.moe_mlp_hidden_dim,
                channel_identity_dim=configs.channel_identity_dim,
                use_softmax=configs.use_softmax,
                grouped_bias=configs.grouped_bias
                )

        self.temporal = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.d_model),
        )
        
        if self.inout_type == 'in':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.inout_type == 'out':
            self.projection = HeEmb(
                n = configs.enc_in, 
                input_dim=configs.d_model, 
                output_dim = configs.pred_len, 
                model_type = configs.model_type,
                rank=configs.rank,
                num_experts=configs.num_experts,
                moe_router_type=configs.moe_router_type,
                moe_mlp_hidden_dim=configs.moe_mlp_hidden_dim,
                channel_identity_dim=configs.channel_identity_dim,
                use_softmax=configs.use_softmax,
                grouped_bias=configs.grouped_bias
                )
        elif self.inout_type == 'inout':
            self.projection = HeEmb(
                n = configs.enc_in, 
                input_dim=configs.d_model, 
                output_dim = configs.pred_len, 
                model_type = configs.model_type,
                rank=configs.rank,
                num_experts=configs.num_experts,
                moe_router_type=configs.moe_router_type,
                moe_mlp_hidden_dim=configs.moe_mlp_hidden_dim,
                channel_identity_dim=configs.channel_identity_dim,
                use_softmax=configs.use_softmax,
                grouped_bias=configs.grouped_bias
                )

    def forward(self, x):
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-6
            x = (x - mean) / std

        x = x.permute(0,2,1)

        x = self.emb(x)
        x = self.temporal(x) + x
        pred = self.projection(x)

        pred = pred.permute(0,2,1)

        if self.use_revin:
            pred = pred * std + mean

        return pred
        
