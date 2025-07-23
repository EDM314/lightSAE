import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='SparseTSF & other models for Time Series Forecasting')
parser.add_argument('--smoke_test', action='store_true', default=False,
                        help='Run in smoke test mode (1 epoch, 1 batch per epoch)')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='SparseTSF', help='model name')
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--force_refresh_cache', action='store_true', default=False,
                    help='Force refresh the cache.')
parser.add_argument('--use_data_cache', action='store_true', default=False,
                    help='Enable loading and saving of preprocessed dataset objects to speed up startup.')
parser.add_argument('--data_cache_path', type=str, default='./datacache/',
                    help='Directory to store cached dataset objects.')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--inout_type', default='in', help='in,out,inout')
parser.add_argument('--model_type', default='SharedEmb', help='see HeEmb')
parser.add_argument('--rank', type=int, default=8, help='rank')
parser.add_argument('--num_experts', type=int, default=10, help='number of experts')
parser.add_argument('--moe_mlp_hidden_dim', type=int, default=512, help='hidden dimension of MoE MLP')
parser.add_argument('--channel_identity_dim', type=int, default=32, help='identity dimension of MoE')
parser.add_argument('--moe_router_type', type=str, default='learned', help='type of MoE router, options: [learned, mlp]')
parser.add_argument('--use_softmax', type=int, default=1, help='use softmax for MoE router, 1: use, 0: not use')
parser.add_argument('--grouped_bias', type=int, default=0, help='use grouped bias for MoE router, 1: use, 0: not use')
parser.add_argument('--period_len', type=int, default=24, help='period length')
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0') 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', 
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='max norm of the gradients for clipping, 0 means no clipping')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus', default=0)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args, unknown = parser.parse_known_args()
if unknown:
    print(f"Unknown arguments received and ignored: {unknown}")
fix_seed_list = range(2023, 2033)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

def get_setting_string(args, iteration, seed):
    return (
        f'{args.model_id}_'
        f'{args.model}_'
        f'{args.data}_'
        f'ft{args.features}_'
        f'sl{args.seq_len}_'
        f'pl{args.pred_len}_'
        f'{args.model_type}_'
        f'rank_{args.rank}_'
        f'num_experts_{args.num_experts}_'
        f'moe_dim_{args.moe_mlp_hidden_dim}_'
        f'cid_{args.channel_identity_dim}_'
        f'router_{args.moe_router_type}_'
        f'{args.des}_'
        f'{iteration}_'
        f'seed{seed}_'
        f'smoke_test_{args.smoke_test}_'
        f'lr_{args.learning_rate}_'
    )


if args.is_training:
    for ii in range(args.itr):
        seed = fix_seed_list[ii]
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        setting = get_setting_string(args, ii, seed)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    seed = fix_seed_list[ii]
    setting = get_setting_string(args, ii, seed)

    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
