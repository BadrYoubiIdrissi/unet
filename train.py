import os
from numpy.core.numeric import indices
from pytorch_lightning import callbacks

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import optimizer
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.metrics.functional import accuracy

from unet1d_model import UNet
from torchsummary import summary
import torchmetrics
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import argparse
import h5py
import pandas as pd
import os
import numpy as np
import tqdm
import hydra
from metric_dreem import dreem_sleep_apnea_custom_metric
import logging
from dice_loss import dice_loss
from omegaconf import OmegaConf

class DreemDataset(LightningDataModule):
    def __init__(self, *args, h5_path=None, h5_test=None, target_path=None, batch_size=None, means=None, stds=None, **kwargs):
        self.h5_path = h5_path
        self.h5_test = h5_test
        self.target_path = target_path
        self.batch_size = batch_size
        self.means = means
        self.stds = stds
        super().__init__()

    def load_data(self, path):
        input_h5 = h5py.File(path)
        
        signals = ['abdom_belt','airflow','PPG','thorac_belt','snore','SPO2','C4-A1','O2-A1']

        data = input_h5["data"] #(batch, all_cols)
        all_data = []
        for j, signal in enumerate(signals):
            col_slice = slice(2+j*9000, 2+(j+1)*9000)
            all_data.append(torch.Tensor(data[:, col_slice]))
        all_data = torch.stack(all_data, dim=1)
        return all_data

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # (batch, 8, 9000)
        self.all_data = self.load_data(self.h5_path)
        print(self.all_data.shape)
        self.test_data = self.load_data(self.h5_test)
        if self.target_path is not None:
            # self.input_labels = torch.Tensor(np.repeat(pd.read_csv(self.target_path, index_col="ID").values, 100, axis=1)).unsqueeze(1)
            self.input_labels = torch.Tensor(pd.read_csv(self.target_path, index_col="ID").values).unsqueeze(1)
            # assert self.input_labels.shape[-1] == 9000
        
        if self.means is None:
          train_stats = self.all_data.permute(1, 0, 2).contiguous()
          train_stats = train_stats[:,:3600].view(8, -1)
          self.means = train_stats.mean(dim=1)
          self.stds = train_stats.std(dim=1)

        self.means = self.means.view(1,8,1)
        self.stds = self.stds.view(1,8,1)
        self.all_data = (self.all_data - self.means)/(self.stds+1e-10)
        self.test_data = (self.test_data - self.means)/(self.stds+1e-10)
        logging.info(f"Dataset means :{self.means.view(-1)}")
        logging.info(f"Dataset stds :{self.stds.view(-1)}")
        
    def setup(self):
        self.train_dataset = TensorDataset(self.all_data[:3600], self.input_labels[:3600])
        self.val_dataset = TensorDataset(self.all_data[3600:], self.input_labels[3600:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=8, batch_sampler=BalancedBatchSampler(self.batch_size, self.all_data[:3600], self.input_labels[:3600]))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)



class SegmentationCallback(Callback):
    def __init__(self, name, sample, target, sig_id, width, height, period) -> None:
        super().__init__()
        self.name=name
        self.sample=sample
        self.target=target
        self.sig_id=sig_id
        self.nb_samples=sample.size(0)
        self.width=width
        self.height=height
        assert width*height == self.nb_samples
        self.counter = 0
        self.period=period
    def on_epoch_end(self, trainer, pl_module):
        # nb_signals = 4
        # pl_module.eval()
        self.counter += 1
        if (self.counter%self.period)==0:
          count = 0
          out = pl_module.model(self.sample)
          pred = out > 0

          fig = plt.figure()
          plt.hist(out.view(-1).detach().cpu().numpy(), bins="auto")
          wandb.log({"hist_predictions_"+self.name: wandb.Image(fig)}, commit=False)
          plt.close()

          fig, axs = plt.subplots(self.height,self.width, sharex=True, squeeze=True, figsize=(45,15))

          with torch.no_grad():
              for i in range(self.height):
                  for j in range(self.width):
                      data_sample, labels = self.sample[count,self.sig_id], self.target[count].squeeze()
                      curr_pred = pred[count].squeeze()
                      count+=1
                      x=np.linspace(0, 90, num=9000)
                      axs[i,j].plot(x, data_sample.detach().cpu().numpy())
                      axs[i,j].fill_between(x, 0, 1, where=np.repeat(labels.bool().detach().cpu().numpy(),100,axis=-1),
                      facecolor='green', linewidth=0.0, alpha=0.2, transform=axs[i,j].get_xaxis_transform())

                      axs[i,j].fill_between(x, 0, 1, where=np.repeat(curr_pred.bool().detach().cpu().numpy(),100,axis=-1),
                      facecolor='red', linewidth=0.0, alpha=0.2, transform=axs[i,j].get_xaxis_transform())

              wandb.log({"sample_predictions_"+self.name: wandb.Image(fig)}, commit=False)
          plt.close()
          # pl_module.train()

def long_mask_to_short_mask(out):
    return (torch.nn.functional.conv1d(out.float(), torch.ones(1, 1, 100, device=out.device), bias=None, stride=100).squeeze() > 50).long().detach().cpu()

class BalancedBatchSampler(object):
    def __init__(self, batch_size, data, target):
        self.all_idx = torch.arange(0, data.size(0))
        self.idx_pos = torch.masked_select(self.all_idx, target.sum(dim=-1).squeeze()>0)
        self.idx_neg = torch.masked_select(self.all_idx, target.sum(dim=-1).squeeze()==0)
        self.half = batch_size//2
        self.len = data.size(0)//batch_size
        self.batch_size=batch_size

    def __len__(self): 
        return self.len

    def __iter__(self):
        for i in range(self.len):
            idx_pos_batch = self.idx_pos[torch.randint(0, self.idx_pos.size(0), (self.half,))]
            idx_neg_batch = self.idx_neg[torch.randint(0, self.idx_pos.size(0), (self.batch_size-self.half,))]
            yield torch.cat([idx_pos_batch, idx_neg_batch], axis=0).view(-1)
    
class DreemUnetModule(pl.LightningModule):

    def __init__(self, model_params, optimizer_params, class_weight=1.0):
        super().__init__()
        unet = UNet(n_channels=8, **model_params)
        self.model = nn.Sequential(unet, nn.Conv1d(1, 1, kernel_size=100, stride=100, pad=1))
        self.optimizer_params = optimizer_params
        self.class_weight = class_weight
        print(summary(self.model, (8, 9000)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x).sigmoid()
        # z = self.model(x)
        # loss = F.binary_cross_entropy_with_logits(z.view(-1), y.view(-1), pos_weight=self.class_weight)
        loss = 1-dice_loss(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x).sigmoid()
        # z = self.model(x)
        # loss = F.binary_cross_entropy_with_logits(z.view(-1), y.view(-1), pos_weight=self.class_weight)
        loss = 1-dice_loss(z, y)
        self.log('val_loss', loss)
        pred = (z>0.8).long()
        self.log('val_f1_score', torchmetrics.functional.f1(pred.view(-1), y.view(-1).long(), num_classes=2, average="none")[1], on_epoch=True, prog_bar=True)
        # self.log('val_dreem_met', dreem_sleep_apnea_custom_metric(pred.long().squeeze().detach().cpu(), y.squeeze().long().detach().cpu()), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_prec_score', torchmetrics.functional.precision(pred.view(-1), y.view(-1).long(), num_classes=2, average="none")[1], on_epoch=True, prog_bar=True)
        self.log('val_recall_score', torchmetrics.functional.recall(pred.view(-1), y.view(-1).long(), num_classes=2, average="none")[1], on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        z = self.model(batch).sigmoid()
        return z
    
    def test_epoch_end(self, outputs):
        self.test_out = torch.cat(outputs)
        np.save("test.npy", self.test_out.detach().cpu().numpy())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return {
          'optimizer': optimizer,
          'lr_scheduler': torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.75, step_size=2)
        }

@hydra.main(config_name="config")
def app(cfg):
    print("Processing dataset")
    dataset = DreemDataset(**cfg.data)
    dataset.prepare_data()
    dataset.setup()
    all_labels = dataset.input_labels.numel()
    pos_labels = dataset.input_labels.sum()
    neg_labels = all_labels-pos_labels
    class_weight = neg_labels/pos_labels
    print(f"All labels: {all_labels}, Positive labels: {pos_labels}, Negative Labels: {neg_labels}")

    module = DreemUnetModule(cfg.model, cfg.optimizer, class_weight=cfg.loss.balancing*class_weight)
    
    wandb_logger = pl.loggers.wandb.WandbLogger(project="dreem_challenge", config=OmegaConf.to_container(cfg, resolve=True))
    checkpoint = ModelCheckpoint(monitor='val_f1_score', dirpath='saves/')
    
    indices_train = np.random.choice(3600, 8*12)
    train_vis = SegmentationCallback("train", dataset.all_data[indices_train].to(0), dataset.input_labels[indices_train].to(0), 0, 12, 8, 40)
    
    indices_val = 3600+np.random.choice(800, 8*12)
    val_vis = SegmentationCallback("val", dataset.all_data[indices_val].to(0), dataset.input_labels[indices_val].to(0), 0, 12, 8, 40)

    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint, train_vis, val_vis, lr_logger], **cfg.trainer)

    if cfg.job.lr_finder:
        lr_finder = trainer.tuner.lr_find(module, dataset)
        fig = lr_finder.plot(suggest=True)
        wandb_logger.experiment.log({"lr_finder": wandb.Image(fig)}, commit=False)

    # Train the model
    trainer.fit(module, dataset)
    trainer.test(module, dataset)
    return trainer, module, dataset


if __name__ == "__main__":
    app()