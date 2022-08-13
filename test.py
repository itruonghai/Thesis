from monai.networks.nets import SegResNet, SwinUNETR, UNet
from models.SegFormer.SegFormer import SegFormerNet 
from models.SegTransVAE.SegTransVAE import SegTransVAE
import monai
import torch
from torchprofile import profile_macs
model = {
    'SegTransVAE': SegTransVAE((128, 128, 128), 
                               8, 4, 3, 768, 
                               8, 4, 3072, 
                               use_VAE = False), 
    'S3DFormer_Attention': SegFormerNet(3, [128, 128, 128], 
                                        in_chans = 4, mixer_type = ['attention', 'attention', 'attention', 'attention']),
    'S3DFormer_Pooling': SegFormerNet(3, [128, 128, 128], 
                                      in_chans = 4, mixer_type = ['pooling', 'pooling', 'pooling', 'pooling']),
    'S3DFormer_Conv': SegFormerNet(3, [128, 128, 128], 
                                      in_chans = 4, mixer_type = ['conv', 'conv', 'conv', 'conv']),
    'S3DFormer_MLP': SegFormerNet(3, [128, 128, 128], 
                                      in_chans = 4, mixer_type = ['mlp', 'mlp', 'mlp', 'mlp']),
    'S3DFormer_PoolingAttention': SegFormerNet(3, [128, 128, 128], 
                                      in_chans = 4, mixer_type = ['pooling', 'pooling', 'attention', 'attention']),
    'S3DFormer_ConvAttention': SegFormerNet(3, [128, 128, 128], 
                                      in_chans = 4, mixer_type = ['conv', 'conv', 'attention', 'attention']),
    'S3DFormer_MLPAttention': SegFormerNet(3, [128, 128, 128], 
                                      in_chans = 4, mixer_type = ['mlp', 'mlp', 'attention', 'attention']),
    
    'SwinUNETR': SwinUNETR(   img_size=(128,128,128),
                              in_channels=4,
                              out_channels=3,
                              feature_size=48,), 
    'Unet3D': UNet( spatial_dims=3,
                    in_channels=4,
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                    norm=monai.networks.layers.Norm.BATCH), 
    'SegResNet': SegResNet(
                    blocks_down = [1,2,2,4],
                    blocks_up = [1,1,1],
                    init_filters = 16,
                    in_channels = 4,
                    out_channels = 3, 
                    dropout_prob = 0.2)

}
inputs = torch.randn(1, 4, 128, 128, 128)
for key in model.keys():
    print(key)
    macs = profile_macs(model[key], inputs)
    print(macs)
