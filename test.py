# val_files = [{"image": ["Data_BRATS/BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz", 
#                         "Data_BRATS/BraTS2021_00000/BraTS2021_00000_t1.nii.gz", 
#                         "Data_BRATS/BraTS2021_00000/BraTS2021_00000_t2.nii.gz", 
#                         "Data_BRATS/BraTS2021_00000/BraTS2021_00000_flair.nii.gz"], 
#             "label": "Data_BRATS/BraTS2021_00000/BraTS2021_00000_seg.nii.gz"}]
# from monai.utils import set_determinism
# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     ConvertToMultiChannelBasedOnBratsClassesd,
#     RandSpatialCropd,
#     RandFlipd,
#     MapTransform,
#     NormalizeIntensityd, 
#     RandScaleIntensityd,
#     RandShiftIntensityd,
#     ToTensord,
#     CenterSpatialCropd,
# )
# from monai.data import DataLoader, Dataset
# import numpy as np
# import json
# class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
#     """
#     Convert labels to multi channels based on brats classes:
#     label 1 is the necrotic and non-enhancing tumor core
#     label 2 is the peritumoral edema
#     label 4 is the GD-enhancing tumor
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).

#     """
 
#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             result = []
#             # merge label 1 and label 4 to construct TC
#             result.append(np.logical_or(d[key] == 1, d[key] == 4))
#             # merge labels 1, 2 and 4 to construct WT
#             result.append(
#                 np.logical_or(
#                     np.logical_or(d[key] == 1, d[key] == 4), d[key] == 2
#                 )
#             )
#             # label 4 is ET
#             result.append(d[key] == 4)
#             d[key] = np.stack(result, axis=0).astype(np.float32)
#         return d
# val_transform = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             ConvertToMultiChannelBasedOnBratsClassesd(
#                 keys = ['label']),
#             CenterSpatialCropd(keys=["image", "label"],
#                             roi_size = [128,128,128], 
#                             ),
#             NormalizeIntensityd(keys = "image",
#                                nonzero = True,
#                                channel_wise = True),
#             ToTensord(keys=["image", "label"]),
#         ]
#     )
# val_ds = Dataset(data=val_files, transform=val_transform)
# val_input = val_ds[0]["image"].unsqueeze(0)
# print("Input shape: ", val_input.shape)
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from trainer import BRATS
import os 
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

model = BRATS(return_attn = True)
import torch
weights = torch.load('Epoch 89-MeanDiceScore0.8897.ckpt', map_location='cpu')
model.load_state_dict(weights['state_dict'], strict = True)
a = torch.rand((1,4,128,128,128))
out, attn = model(a)
print(out.shape, len(attn))

