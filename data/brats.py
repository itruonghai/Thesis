from monai.utils import set_determinism
import monai.transforms as transforms 
from monai.data import DataLoader, Dataset
import numpy as np
import json
set_determinism(seed=0)
with open('../brats2021.json') as f:
    data = json.load(f)
train_files, val_files, test_files = data['training'], data['validation'], data['testing']
print(len(train_files), len(val_files), len(test_files))

def get_train_dataloader():

#     train_transform = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             ConvertToMultiChannelBasedOnBratsClassesd(keys = ['label']),
#             # RandSpatialCropd(keys=["image", "label"],
#             #                 roi_size = [128,128,128], 
#             #                 #  roi_size = [96,96,96],
#             #                 random_size = False),
#             CenterSpatialCropd(keys=["image", "label"],
#                             roi_size = [128,128,128], 
#                             ),
#             RandFlipd(keys = ["image", "label"],
#                      prob = 0.5,
#                      spatial_axis = 0),
#             RandFlipd(keys = ["image", "label"],
#                      prob = 0.5,
#                      spatial_axis = 1),
#             RandFlipd(keys = ["image", "label"],
#                      prob = 0.5,
#                      spatial_axis = 2),
#             NormalizeIntensityd(keys = "image",
#                                nonzero = True,
#                                channel_wise = True),
#             RandScaleIntensityd(keys = "image", prob = 1, factors = 0.1),
#             RandShiftIntensityd(keys = "image", prob = 1, offsets = 0.1),
#             ToTensord(keys=["image", "label"]),
#         ]
#     )
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", 
                                       k_divisible=[128, 128, 128]),
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128],
                                        random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ])
    train_ds = Dataset(data=train_files, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8,pin_memory = True)

    
    return train_loader

def get_val_dataloader():
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(
                keys = ['label']),
            transforms.NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory = True)
    
    return val_loader

def get_test_dataloader():
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(
                keys = ['label']),
            transforms.NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_ds = Dataset(data=test_files, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory = True)
    
    return test_loader

