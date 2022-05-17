import torch
import torch.nn as nn
from models.SegFormer.SegFormer import SegFormerNet


def test():
    # x : tensor - [batch_size, channel, w, h]
    x = torch.randn((1, 4, 128, 128, 128))

    # model = ClassicUnet(in_channels=1, out_channels=1)
    model = SegFormerNet(1, [128, 128, 128], in_chans=4)

    pred = model(x)
    # print(x.shape)
    # print(pred.shape)


if __name__ == '__main__':
    test()
