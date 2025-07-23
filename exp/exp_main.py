from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, PatchTST, SparseTSF, iTransformer,RLinear,RLinear_ind,RMLP,RLinear_Emb,iTransformer_Emb,RMLP_Emb,PatchTST_Emb
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop,calculate_he_emb_params
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import wandb
import swanlab

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # Determine if the model uses only batch_x based on the model name
        self.models_using_only_batch_x = {'Linear', 'PatchTST', 'SparseTSF', 'RLinear', 'RLinear_ind','RMLP','RLinear_Emb','RMLP_Emb','PatchTST_Emb'}
        self.use_only_batch_x = self.args.model in self.models_using_only_batch_x

        # Handle smoke_test mode
        if hasattr(self.args, 'smoke_test') and self.args.smoke_test:
            self.args.train_epochs = 1
            print("INFO: Running in SMOKE TEST MODE. train_epochs overridden to 1.")

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SparseTSF': SparseTSF,
            'iTransformer': iTransformer,
            'RLinear': RLinear,
            'RLinear_ind': RLinear_ind,
            'RMLP': RMLP,
            'RLinear_Emb': RLinear_Emb,
            'iTransformer_Emb': iTransformer_Emb,
            'RMLP_Emb':RMLP_Emb,
            'PatchTST_Emb':PatchTST_Emb
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        num_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_size = batch_x.size(0)  # Get current batch size

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.use_only_batch_x:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.use_only_batch_x:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)  # This is the primary loss, usually MSE
                mae = mae_criterion(pred, true)
                mse = mse_criterion(pred, true)

                total_loss += loss.item() * batch_size  # Accumulate (batch avg loss * batch size)
                total_mae += mae.item() * batch_size
                total_mse += mse.item() * batch_size
                num_samples += batch_size  # Accumulate total samples

                if hasattr(self.args, 'smoke_test') and self.args.smoke_test:
                    print(f"INFO: SMOKE TEST MODE - Processed 1 batch in vali. Stopping vali early.")
                    break  # Exit after the first batch

        # Ensure num_samples is not zero to prevent division by zero error
        if num_samples == 0:
            print("WARNING: num_samples is zero in vali method. Returning 0 for losses.")
            avg_loss, avg_mae, avg_mse = 0, 0, 0
        else:
            avg_loss = total_loss / num_samples  # Calculate weighted average loss
            avg_mae = total_mae / num_samples
            avg_mse = total_mse / num_samples
        
        self.model.train()
        return avg_loss, avg_mse, avg_mae

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Initialize swanlab sync (only send to swanlab, not to wandb)
        swanlab.sync_wandb(wandb_run=False)
        
        # Initialize wandb run
        run = wandb.init(
            project="TimeSeries_Experiments", # You can change this to your project name
            name=setting, # Use setting as run name
            group=f"{self.args.model}-{self.args.data}-PL{self.args.pred_len}", # Create group name
            config=vars(self.args) # Record all hyperparameters
        )
        # Record start time to summary and save as instance variable
        start_timestamp = time.time()
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_timestamp))
        self.start_timestamp = start_timestamp # Save as instance variable for test method to use
        wandb.summary["exp/start_timestamp"] = start_timestamp
        wandb.summary["exp/start_time_str"] = start_time_str
        wandb.summary["exp/run_state"] = "running"

        # Save wandb run object in instance so test method can access it
        self.wandb_run = run

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        best_vali_loss = np.inf

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # This learning rate setting is also used in PDF and pathformer, not sure if it originated from this codebase, probably not, since it's not used here, might be for PatchTST
        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        else:
            scheduler = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            total_train_loss = 0.0
            num_train_samples = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_size = batch_x.size(0) # Get current batch size

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.use_only_batch_x:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        total_train_loss += loss.item() * batch_size
                        num_train_samples += batch_size
                else:
                    if self.use_only_batch_x:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    total_train_loss += loss.item() * batch_size
                    num_train_samples += batch_size

                if (i + 1) % 100 == 0:
                    # Can optionally record batch loss, but may generate large amounts of data
                    # wandb.log({
                    #     "train/batch_loss": loss.item(),
                    #     "train/step": epoch * train_steps + i
                    # })
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    # Gradient Clipping for AMP
                    if self.args.clip_grad_norm > 0:
                        scaler.unscale_(model_optim)  # Unscale gradients before clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # Gradient Clipping
                    if self.args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    model_optim.step()

                # Smoke test: break after 1st batch if in smoke_test mode
                if hasattr(self.args, 'smoke_test') and self.args.smoke_test:
                    print(f"INFO: SMOKE TEST MODE - Processed 1 batch in epoch {epoch + 1}. Stopping epoch early.")
                    break # Exit after the first batch

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
                

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = total_train_loss / num_train_samples
            vali_loss, vali_mse, vali_mae = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mse, test_mae = self.vali(test_data, test_loader, criterion) # During training, use vali instead of test method to get loss on test set, test method is used to get predictions on test set and save them as files

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # If better validation performance is found, update best test metrics in wandb summary
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                print(f"New best validation score: {best_vali_loss:.7f}. Updating wandb summary with corresponding test metrics.")
                wandb.summary['best_test_mse'] = test_mse
                wandb.summary['best_test_mae'] = test_mae

            # Record epoch-level metrics to wandb
            wandb.log({
                "train/epoch": epoch + 1,
                "train/train_epoch_loss": train_loss,
                "train/val_epoch_loss": vali_loss,
                "train/test_epoch_loss": test_loss, # Test loss calculated using vali
                "train/lr": model_optim.param_groups[0]['lr'] # Get current learning rate from optimizer
            })

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if hasattr(self.args, 'smoke_test') and self.args.smoke_test:
                print(f"INFO: SMOKE TEST MODE - Epoch {epoch + 1} (1 batch) completed.")

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                # Without this line, learning rate settings don't work when lradj != 'TST'
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0")) # The cuda:0 here and elsewhere in the codebase might be a hidden issue when using multi-GPU, needs modification

        # Save best model to wandb, commented out here because it's found to be a bit wasteful of time and space
        # wandb.save(best_model_path)
        # print(f"Best model saved to {best_model_path} and wandb.")

        # Don't end run here so test can continue recording
        # wandb.finish()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        # Check if there's an active wandb run, if not (e.g., calling test directly), don't record
        wandb_active = hasattr(self, 'wandb_run') and self.wandb_run is not None and self.wandb_run.id is not None

        if test:
            print('loading model')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
            # If loading model for testing and wandb was initialized during train phase, can try to record model path
            # if wandb_active:
                # wandb.config.update({"loaded_model_path": model_path}, allow_val_change=True)

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        total_inference_time = 0.0
        total_samples = 0
        num_batches = len(test_loader)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_start_time = time.time() # Record batch start time
                batch_size = batch_x.size(0) # Get batch size

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.use_only_batch_x:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.use_only_batch_x:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Calculate batch inference time
                batch_inference_time = time.time() - batch_start_time
                total_inference_time += batch_inference_time
                total_samples += batch_size

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if hasattr(self.args, 'smoke_test') and self.args.smoke_test:
                    print(f"INFO: SMOKE TEST MODE - Processed 1 batch in test. Stopping test early.")
                    break # Exit after the first batch
        
        # Ensure num_batches and total_samples are not zero before division, 
        # especially if smoke test caused early exit with only one batch.        
        if num_batches == 0: # Should not happen if loader has data and smoke test processes 1 batch
            avg_inference_time_per_batch = 0
        elif hasattr(self.args, 'smoke_test') and self.args.smoke_test and total_samples > 0 : # If smoke test, num_batches might be len(test_loader) but we only processed 1 actual batch time
             avg_inference_time_per_batch = total_inference_time # total_inference_time is for one batch
        else:
            avg_inference_time_per_batch = total_inference_time / num_batches

        if total_samples == 0: # Should not happen if smoke test processes 1 batch
            avg_inference_time_per_sample = 0
        else:
            avg_inference_time_per_sample = total_inference_time / total_samples

        print(f"Total inference time: {total_inference_time:.4f}s")
        print(f"Avg inference time per batch: {avg_inference_time_per_batch:.4f}s")
        print(f"Avg inference time per sample: {avg_inference_time_per_sample:.6f}s")

        # Calculate FLOPs/MACs and Params by default
        # Calculate FLOPs/MACs and Params by default
        # Use data from the last batch to calculate FLOPs/MACs and Params
        # Ensure tensors are on the correct device
        batch_x = batch_x.to(self.device)
        if not self.use_only_batch_x:
            batch_x_mark = batch_x_mark.to(self.device)
            dec_inp = dec_inp.to(self.device)
            batch_y_mark = batch_y_mark.to(self.device)

        if self.args.test_flop or True:
            print('Calculating MACs and Params...')
            # Prepare model inputs (take first sample from batch to calculate single-sample complexity)
            if self.use_only_batch_x:
                # Only take the first sample
                single_sample_inputs = (batch_x[0:1],)
            else:
                # Ensure all inputs only take the first sample
                single_sample_inputs = (
                    batch_x[0:1],
                    batch_x_mark[0:1],
                    dec_inp[0:1],
                    batch_y_mark[0:1]
                )

            # Call modified function and receive return values (macs_str, params_str, macs_raw, params_raw)
            # Note: This calculates MACs and Params for a single sample
            macs_str, params_str, macs_raw, params_raw, params_manual, params_manual_str = test_params_flop(self.model, inputs=single_sample_inputs)
            
            # Calculate HeEmb module parameters separately
            he_emb_params, he_emb_params_str = calculate_he_emb_params(self.model)

            # Record if wandb is active
            if wandb_active:
                # Record formatted strings and raw values, maintain compatibility with existing fields
                wandb.summary["test/macs"] = macs_str
                wandb.summary["test/params"] = params_str # Parameters calculated by thop
                wandb.summary["test/macs_raw"] = macs_raw
                wandb.summary["test/params_raw"] = params_raw # Raw parameters calculated by thop
                wandb.summary["test/params_manual"] = params_manual
                wandb.summary["test/params_manual_str"] = params_manual_str
                
                # Add HeEmb parameter recording
                wandb.summary["test/he_emb_params"] = he_emb_params
                wandb.summary["test/he_emb_params_str"] = he_emb_params_str
                
                print(f"All metrics logged to wandb.")
            # Remove exit() to continue execution
            # exit()

        # fix bug
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        # Record final test metrics and inference time to wandb summary
        if wandb_active:
            wandb.summary["test/final_mae"] = mae
            wandb.summary["test/final_mse"] = mse
            wandb.summary["test/final_rmse"] = rmse
            wandb.summary["test/final_mape"] = mape
            wandb.summary["test/final_mspe"] = mspe
            wandb.summary["test/final_rse"] = rse
            wandb.summary["test/final_corr"] = corr
            # Add inference time recording
            wandb.summary["test/total_inference_time_s"] = total_inference_time
            wandb.summary["test/avg_inference_time_per_batch_s"] = avg_inference_time_per_batch
            wandb.summary["test/avg_inference_time_per_sample_s"] = avg_inference_time_per_sample
            print("Final test metrics and inference times logged to wandb.")

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # Save metrics and predictions - only enabled in standalone test mode (test=1)
        if test:
            metrics_array = np.array([mae, mse, rmse, mape, mspe, rse, corr], dtype=object)
            np.save(folder_path + 'metrics.npy', metrics_array)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'input.npy', inputx)
            print(f"Predictions saved to {folder_path}pred.npy")
            print(f"Ground truth saved to {folder_path}true.npy")
            print(f"Input data saved to {folder_path}input.npy")

        # Save prediction results and ground truth, and record as wandb artifact
        # pred_path = folder_path + 'pred.npy'
        # true_path = folder_path + 'true.npy'
        # np.save(pred_path, preds)
        # np.save(true_path, trues)
        # print(f"Predictions saved to {pred_path}")
        # print(f"Ground truth saved to {true_path}")

        # if wandb_active:
        #     predictions_artifact = wandb.Artifact(f"run_{self.wandb_run.id}_predictions", type="predictions")
        #     predictions_artifact.add_file(pred_path)
        #     predictions_artifact.add_file(true_path)
        #     self.wandb_run.log_artifact(predictions_artifact)
        #     print("Prediction artifacts logged to wandb.")

        # np.save(folder_path + 'x.npy', inputx)

        # End wandb run after test finishes
        if wandb_active:
            # Record end time and run duration
            end_timestamp = time.time()
            end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_timestamp))
            duration_seconds = None
            # Get start timestamp from instance variable
            if hasattr(self, 'start_timestamp') and self.start_timestamp:
                 duration_seconds = end_timestamp - self.start_timestamp
                 wandb.summary["exp/run_duration_seconds"] = duration_seconds

            wandb.summary["exp/end_timestamp"] = end_timestamp
            wandb.summary["exp/end_time_str"] = end_time_str
            wandb.summary["exp/run_state"] = "finished"

            self.wandb_run.finish()
            print("Wandb run finished.")

        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.use_only_batch_x:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.use_only_batch_x:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
