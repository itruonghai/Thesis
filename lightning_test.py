from data.brats import get_train_dataloader, get_val_dataloader, get_test_dataloader
import pytorch_lightning as pl
from trainer import BRATS
import os 
import torch
import argparse

os.system('cls||clear')
print("Testing ...")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--ckpt', type=str, default = None)
parser.add_argument('--gpu')

args = parser.parse_args()
model = BRATS(args.model)
if args.ckpt != 'None':
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
model.eval()
val_dataloader = get_val_dataloader()
test_dataloader = get_test_dataloader()
trainer = pl.Trainer(gpus = [int(args.gpu)], precision=16, progress_bar_refresh_rate=10)

trainer.test(model, dataloaders = val_dataloader)
# trainer.test(model, dataloaders = test_dataloader)

