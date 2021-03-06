# Torch Library
import torch
import torch.nn.functional as F
import torch.nn as nn 

# MONAI
from monai.networks.nets import SegResNet, UNETR
from monai.metrics import DiceMetric#, compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
import sys
# insert at 1, 0 is the script path (or '' in REPL)
from models.SegFormer.SegFormer import SegFormerNet

# Pytorch Lightning
import pytorch_lightning as pl

# Custom Libraries
# from models.SegTransVAE import SegTransVAE
from data.brats import get_train_dataloader, get_val_dataloader, get_test_dataloader

from loss.loss import DiceScore, Loss_VAE
# from adabelief_pytorch import AdaBelief
import matplotlib.pyplot as plt

# from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
import csv
import os

class BRATS(pl.LightningModule):
    def __init__(self, return_attn = False, ):
        super().__init__()
     
        # self.model = SegTransVAE((128, 128, 128), 8, 4, 3, 768, 8, 4, 3072, use_VAE = use_VAE)
        self.model = SegFormerNet(3, [128, 128, 128], in_chans = 4)
        # self.model = SegResNet(
        #         blocks_down = [1,2,2,4],
        #         blocks_up = [1,1,1],
        #         init_filters = 16,
        #         in_channels = 4,
        #         out_channels = 3, 
        #         dropout_prob = 0.2
        #         )
        self.loss_vae = Loss_VAE()
        self.dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        self.post_trans_images = Compose(
                [EnsureType(),
                 Activations(sigmoid=True), 
                 AsDiscrete(threshold_values=True), 
                 ]
            )
        self.return_attn = return_attn
        self.best_val_dice = 0
    def forward(self, x):
        return self.model(x, self.return_attn) 
    def training_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
      
        outputs = self.forward(inputs)
        loss = self.dice_loss(outputs, labels)
        

        self.log('train/loss', loss)
        
        return loss
    def validation_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        roi_size = (128, 128, 128)
        sw_batch_size = 1
        outputs = sliding_window_inference(
                inputs, roi_size, sw_batch_size, self.model, overlap = 0.5)
        loss = self.dice_loss(outputs, labels)
        
      
        val_outputs = self.post_trans_images(outputs)
        metric_tc = DiceScore(y_pred=val_outputs[:, 0:1], y=labels[:, 0:1], include_background = True)
        metric_wt = DiceScore(y_pred=val_outputs[:, 1:2], y=labels[:, 1:2], include_background = True)
        metric_et = DiceScore(y_pred=val_outputs[:, 2:3], y=labels[:, 2:3], include_background = True)
        mean_val_dice =  (metric_tc + metric_wt + metric_et)/3
        return {'val_loss': loss, 'val_mean_dice': mean_val_dice, 'val_dice_tc': metric_tc,
                'val_dice_wt': metric_wt, 'val_dice_et': metric_et}
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_val_dice = torch.stack([x['val_mean_dice'] for x in outputs]).mean()
        metric_tc = torch.stack([x['val_dice_tc'] for x in outputs]).mean()
        metric_wt = torch.stack([x['val_dice_wt'] for x in outputs]).mean()
        metric_et = torch.stack([x['val_dice_et'] for x in outputs]).mean()
        self.log('val/Loss', loss)
        self.log('val/MeanDiceScore', mean_val_dice)
        self.log('val/DiceTC', metric_tc)
        self.log('val/DiceWT', metric_wt)
        self.log('val/DiceET', metric_et)
        os.makedirs(self.logger.log_dir,  exist_ok=True)
        if self.current_epoch == 0:
            with open('{}/metric_log.csv'.format(self.logger.log_dir), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Mean Dice Score', 'Dice TC', 'Dice WT', 'Dice ET'])
        with open('{}/metric_log.csv'.format(self.logger.log_dir), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_epoch, mean_val_dice.item(), metric_tc.item(), metric_wt.item(), metric_et.item()])

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
                f"\n Current epoch: {self.current_epoch} Current mean dice: {mean_val_dice:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\n Best mean dice: {self.best_val_dice}"
                f" at epoch: {self.best_val_epoch}"
            )
        return {'val_MeanDiceScore': mean_val_dice}
    def test_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
    
        roi_size = (128, 128, 128)
        sw_batch_size = 1
        test_outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, self.forward, overlap = 0.5)
        loss = self.dice_loss(test_outputs, labels)
        test_outputs = self.post_trans_images(test_outputs)
        metric_tc = DiceScore(y_pred=test_outputs[:, 0:1], y=labels[:, 0:1], include_background = True)
        metric_wt = DiceScore(y_pred=test_outputs[:, 1:2], y=labels[:, 1:2], include_background = True)
        metric_et = DiceScore(y_pred=test_outputs[:, 2:3], y=labels[:, 2:3], include_background = True)
        mean_test_dice =  (metric_tc + metric_wt + metric_et)/3
        return {'test_loss': loss, 'test_mean_dice': mean_test_dice, 'test_dice_tc': metric_tc,
                'test_dice_wt': metric_wt, 'test_dice_et': metric_et}
    
    def test_epoch_end(self, outputs):
        loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        mean_test_dice = torch.stack([x['test_mean_dice'] for x in outputs]).mean()
        metric_tc = torch.stack([x['test_dice_tc'] for x in outputs]).mean()
        metric_wt = torch.stack([x['test_dice_wt'] for x in outputs]).mean()
        metric_et = torch.stack([x['test_dice_et'] for x in outputs]).mean()
        self.log('test/Loss', loss)
        self.log('test/MeanDiceScore', mean_test_dice)
        self.log('test/DiceTC', metric_tc)
        self.log('test/DiceWT', metric_wt)
        self.log('test/DiceET', metric_et)

        with open('{}/test_log.csv'.format(self.logger.log_dir), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Mean Test Dice", "Dice TC", "Dice WT", "Dice ET"])
            writer.writerow([mean_test_dice, metric_tc, metric_wt, metric_et])

        return {'test_MeanDiceScore': mean_test_dice}
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                    self.model.parameters(), self.lr, weight_decay=1e-5, amsgrad=True
                    )
#         optimizer = AdaBelief(self.model.parameters(), 
#                             lr=self.lr, eps=1e-16, 
#                             betas=(0.9,0.999), weight_decouple = True, 
#                             rectify = False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return get_train_dataloader()
    
    def val_dataloader(self):
        return get_val_dataloader()
    
    def test_dataloader(self):
        return get_test_dataloader()
    
   