from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from trainer import BRATS
import os 
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import torch
import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
    
model = BRATS()
# weights = torch.load('Epoch 89-MeanDiceScore0.8897.ckpt', map_location='cpu')
# model.load_state_dict(weights['state_dict'], strict = True)
# out, attn = model(val_input)
# print(out.shape, len(attn))
cam = GradCAM(model=model,
            target_layers=[model.model.backbone.norm4],
            use_cuda=False,
            reshape_transform=None)
output = torch.rand(3, 128, 128 , 128)
targets = [output[2, :, :, :].sum()]
grayscale_cam = cam(input_tensor=torch.rand(1,4,128,128,128),
                    targets=targets ,
                    eigen_smooth=False,
                    aug_smooth=False)
print(grayscale_cam.shape)
