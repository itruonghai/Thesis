from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from trainer import BRATS
import os 
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import torch 
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--gpu')
args = parser.parse_args()

os.system('cls||clear')
print("Training ...")
model = BRATS(args.model)
checkpoint_callback = ModelCheckpoint(
    monitor='val/MeanDiceScore',
    dirpath='./ckpt/{}'.format(args.exp),
    save_top_k= 5, 
    filename='Epoch{epoch:3d}-MeanDiceScore{val/MeanDiceScore:.4f}',
    every_n_epochs = 10,
    mode='max',
    save_last= True,
    auto_insert_metric_name=False
)
early_stop_callback = EarlyStopping(
   monitor='val/MeanDiceScore',
   min_delta=0.0001,
   patience=15,
   verbose=False,
   mode='max'
)
tensorboardlogger = TensorBoardLogger(
    'logs', 
    name = args.exp, 
    default_hp_metric = None 
)
trainer = pl.Trainer(#fast_dev_run = 10, 
#                     accelerator='ddp',
                    #overfit_batches=5,
                     gpus = [int(args.gpu)], 
                        precision=16,
                     max_epochs = 200, 
                     progress_bar_refresh_rate=10,  
                     callbacks=[checkpoint_callback, early_stop_callback], 
#                     auto_lr_find=True,
                    num_sanity_val_steps=0,
                    logger = tensorboardlogger,
#                     limit_train_batches=0.01, 
#                     limit_val_batches=0.01
                     )
# trainer.tune(model)
trainer.fit(model)



