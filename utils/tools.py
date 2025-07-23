import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

# Import thop
from thop import profile, clever_format

def test_params_flop(model, inputs):
    """
    Calculates MACs and Params using thop, supporting multiple inputs.
    Args:
        model (torch.nn.Module): The model to analyze.
        inputs (tuple): A tuple containing the input tensors for the model's forward pass.
                        For single-input models: (batch_x,)
                        For multi-input models: (batch_x, batch_x_mark, dec_inp, batch_y_mark)
    Returns:
        tuple: (macs_str, params_str, macs_raw, params_raw)
               - macs_str (str): Formatted MACs string (e.g., "1.23 GMac").
               - params_str (str): Formatted Params string (e.g., "10.5 M").
               - macs_raw (float): Raw MACs count.
               - params_raw (float): Raw Params count.
    """
    # Ensure model and inputs are on the same device, thop suggests analyzing on CPU for more consistent results
    # But to maintain device consistency with the original code, we temporarily don't move devices internally
    # device = next(model.parameters()).device
    # inputs_on_device = tuple(inp.to(device) for inp in inputs)
    # model.to(device) # Ensure model is also on the correct device

    # Use thop to calculate MACs and Params
    # Note: If model or inputs are not on CPU, thop may give warnings or be inaccurate on some CUDA operations
    # custom_ops is used to handle custom operations that may exist in the model, temporarily empty here
    macs_raw, params_raw = profile(model, inputs=inputs, verbose=False)

    # Use clever_format to format output
    macs_str, params_str = clever_format([macs_raw, params_raw], "%.3f")

    print('{:<30}  {:<8}'.format('Computational complexity (MACs):', macs_str))
    print('{:<30}  {:<8}'.format('Number of parameters (thop):', params_str))

    # Manually calculate parameters
    params_manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_manual_str = clever_format([params_manual], "%.3f")
    print('{:<30}  {:<8}'.format('Number of parameters (manual):', params_manual_str))

    # Return formatted strings and raw values
    return macs_str, params_str, macs_raw, params_raw, params_manual, params_manual_str

def calculate_he_emb_params(model):
    """
    Calculates the trainable parameters of all HeEmb modules within a model.
    Args:
        model (torch.nn.Module): The model to analyze.
    Returns:
        tuple: (he_emb_params, he_emb_params_str)
               - he_emb_params (int): Raw count of HeEmb parameters.
               - he_emb_params_str (str): Formatted HeEmb parameters string.
    """
    from layers.HeEmb import HeEmb # Local import to avoid circular dependencies or unnecessary global imports
    
    he_emb_params = 0
    for module in model.modules():
        if isinstance(module, HeEmb):
            he_emb_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    he_emb_params_str = clever_format([he_emb_params], "%.3f")
    print('{:<30}  {:<8}'.format('HeEmb parameters (manual):', he_emb_params_str))
    return he_emb_params, he_emb_params_str