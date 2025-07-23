import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import numpy as np
from layers.HeEmb import HeEmb

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,enc_in=None,embed_layer=None):
        super(DataEmbedding_inverted, self).__init__()

        if embed_layer == None:
            self.value_embedding = nn.Linear(c_in, d_model)
        else:
            assert isinstance(embed_layer,nn.Module)
            self.value_embedding = embed_layer

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
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
            

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, 
            configs.d_model, 
            configs.embed, 
            configs.freq, 
            configs.dropout,
            configs.enc_in,
            embed_layer= self.emb
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        if self.inout_type == 'in':
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.inout_type == 'out':
            self.projector = HeEmb(
                n = configs.enc_in, 
                input_dim= configs.d_model, 
                output_dim = configs.pred_len, 
                model_type = configs.model_type,
                rank=configs.rank,
                num_experts=configs.num_experts,
                moe_router_type=configs.moe_router_type,
                moe_mlp_hidden_dim=configs.moe_mlp_hidden_dim,
                channel_identity_dim=configs.channel_identity_dim
                )
        elif self.inout_type == 'inout':
            self.projector = HeEmb(
                n = configs.enc_in, 
                input_dim= configs.d_model, 
                output_dim = configs.pred_len, 
                model_type = configs.model_type,
                rank=configs.rank,
                num_experts=configs.num_experts,
                moe_router_type=configs.moe_router_type,
                moe_mlp_hidden_dim=configs.moe_mlp_hidden_dim,
                channel_identity_dim=configs.channel_identity_dim
                )
            


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_revin:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None)
        
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_revin:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]