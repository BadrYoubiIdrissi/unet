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

from pytorch_lightning.callbacks import ModelCheckpoint
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
    def __init__(self, *args, h5_path=None, target_path=None, batch_size=None, **kwargs):
        self.h5_path = h5_path
        self.target_path = target_path
        self.batch_size = batch_size
        super().__init__()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # (batch, 8, 9000)
        input_h5 = h5py.File(self.h5_path)
        if self.target_path is not None:
            self.input_labels = torch.Tensor(np.repeat(pd.read_csv(self.target_path, index_col="ID").values, 100, axis=1)).unsqueeze(1)
            # self.input_labels = torch.Tensor(pd.read_csv(self.target_path, index_col="ID").values).unsqueeze(1)
            # assert self.input_labels.shape[-1] == 9000
        signals = ['abdom_belt','airflow','PPG','thorac_belt','snore','SPO2','C4-A1','O2-A1']

        data = input_h5["data"] #(batch, all_cols)
        all_data = []
        for j, signal in enumerate(signals):
            col_slice = slice(2+j*9000, 2+(j+1)*9000)
            all_data.append(torch.Tensor(data[:, col_slice]))
        self.all_data = torch.stack(all_data, dim=1)[:,:6,:]
        # train_stats = self.all_data[:,3600].view(8, -1)
        # self.mean = train_stats.mean(dim=1).view(1,8,1)
        # self.std = train_stats.std(dim=1).view(1,8,1)
        # self.all_data = (self.all_data.permute(1, 0, 2) - self.mean)/(self.std+1e-10)
        # logging.info(f"Dataset means :{self.mean.view(-1)}")
        # logging.info(f"Dataset stds :{self.std.view(-1)}")
        
    def setup(self):
        self.train_dataset = TensorDataset(self.all_data[:3600], self.input_labels[:3600])
        self.val_dataset = TensorDataset(self.all_data[3600:], self.input_labels[3600:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16, shuffle=False)



class SegmentationCallback(Callback):
    def __init__(self, name, sample, target, sig_id, width, height) -> None:
        super().__init__()
        self.name=name
        self.sample=sample
        self.target=target
        self.sig_id=sig_id
        self.nb_samples=sample.size(0)
        self.width=width
        self.height=height
        assert width*height == self.nb_samples
    def on_epoch_end(self, trainer, pl_module):
        # nb_signals = 4
        # pl_module.eval()
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
                    axs[i,j].fill_between(x, 0, 1, where=labels.bool().detach().cpu().numpy(),
                    facecolor='green', linewidth=0.0, alpha=0.2, transform=axs[i,j].get_xaxis_transform())

                    axs[i,j].fill_between(x, 0, 1, where=curr_pred.bool().detach().cpu().numpy(),
                    facecolor='red', linewidth=0.0, alpha=0.2, transform=axs[i,j].get_xaxis_transform())

            wandb.log({"sample_predictions_"+self.name: wandb.Image(fig)}, commit=False)
        plt.close()
        # pl_module.train()

# def dice_loss(input, target):
#     smooth = 1e-3

#     iflat = input.view(-1)
#     tflat = target.view(-1)
#     intersection = (iflat * tflat).sum()
    
#     return 1 - ((2. * intersection + smooth) /
#               (iflat.sum() + tflat.sum() + smooth))

def long_mask_to_short_mask(out):
    return (torch.nn.functional.conv1d(out.float(), torch.ones(1, 1, 100, device=out.device), bias=None, stride=100).squeeze() > 50).long().detach().cpu()

class DreemUnetModule(pl.LightningModule):

    def __init__(self, model_params, optimizer_params, class_weight=1.0):
        super().__init__()
        # down_sample = nn.Sequential(nn.Conv1d(8, 32, 7, padding=9, dilation=3),
        #                                  nn.ReLU(), 
        #                                  nn.MaxPool1d(5), 
        #                                  nn.Conv1d(32, 64, 7, padding=9, dilation=3), 
        #                                  nn.ReLU(), 
        #                                  nn.MaxPool1d(2),
        #                                  nn.Conv1d(64, 64, 7, padding=9, dilation=3), 
        #                                  nn.ReLU(), 
        #                                  nn.MaxPool1d(5), 
        #                                  nn.Conv1d(64, 64, 7, padding=9, dilation=3), 
        #                                  nn.ReLU(), 
        #                                  nn.MaxPool1d(2))
        unet = UNet(n_channels=6, **model_params)
        self.model = unet
        self.optimizer_params = optimizer_params
        self.class_weight = class_weight
        print(summary(self.model, (6, 9000)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x).sigmoid()
        # loss = F.binary_cross_entropy_with_logits(z.view(-1), y.view(-1), weight=self.class_weight)
        loss = 1-dice_loss(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x).sigmoid()
        # loss = F.binary_cross_entropy_with_logits(z.view(-1), y.view(-1), weight=self.class_weight)
        loss = 1-dice_loss(z, y)
        self.log('val_loss', loss)
        pred = (z>0.5).long()
        self.log('val_f1_score', torchmetrics.functional.f1(pred.view(-1), y.view(-1).long(), num_classes=2, average="none")[1], on_step=True, on_epoch=True, prog_bar=True)
        # self.log('val_dreem_met', dreem_sleep_apnea_custom_metric(pred.long().squeeze().detach().cpu(), y.squeeze().long().detach().cpu()), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_prec_score', torchmetrics.functional.precision(pred.view(-1), y.view(-1).long(), num_classes=2, average="none")[1], on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_recall_score', torchmetrics.functional.recall(pred.view(-1), y.view(-1).long(), num_classes=2, average="none")[1], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = F.binary_cross_entropy_with_logits(z.view(-1), y.view(-1), weight=self.class_weight)
        self.log('test_loss', loss)
        pred = (z.squeeze()>0).long()
        self.log('test_f1_score', torchmetrics.functional.f1(pred.view(-1), y.view(-1), 2, average="none")[1], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_prec_score', torchmetrics.functional.precision(pred.view(-1), y.view(-1), average="none")[1], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall_score', torchmetrics.functional.recall(pred.view(-1), y.view(-1), average="none")[1], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer
        # {
        #         'optimizer': optimizer,
        #         # 'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2),
        #         # 'monitor': 'val_loss'
        #     }

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
    checkpoint = ModelCheckpoint(monitor='val_loss')
    
    indices_train = np.random.choice(3600, 8*12)
    train_vis = SegmentationCallback("train", dataset.all_data[indices_train].to(0), dataset.input_labels[indices_train].to(0), 0, 12, 8)
    
    indices_val = 3600+np.random.choice(800, 8*12)
    val_vis = SegmentationCallback("val", dataset.all_data[indices_val].to(0), dataset.input_labels[indices_val].to(0), 0, 12, 8)

    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint, train_vis, val_vis], **cfg.trainer)
    
    if cfg.job.lr_finder:
        lr_finder = trainer.tuner.lr_find(module, dataset)
        fig = lr_finder.plot(suggest=True)
        wandb_logger.experiment.log({"lr_finder": wandb.Image(fig)}, commit=False)

    # Train the model
    trainer.fit(module, dataset)


if __name__ == "__main__":
    app()