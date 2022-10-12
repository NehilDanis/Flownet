import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np


def conv(in_channel, out_channel, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=padding),
        nn.ReLU()
    )


def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()


class FlowNetSimple(pl.LightningModule):

    def __init__(self):
        super().__init__()
        '''
        One could use sequential here too, but since we will be needing the output of some of the conv layers in encoder
        , as an input to decoder layers, it is better to specify each layer seperately.
        '''
        self.conv1 = conv(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = conv(256, 256)
        self.conv4 = conv(256, 512, stride=2, padding=1)
        self.conv4_1 = conv(512, 512)
        self.conv5 = conv(512, 512, stride=2, padding=1)
        self.conv5_1 = conv(512, 512)
        self.conv6 = conv(512, 1024, stride=2, padding=1)

    def forward(self, imgs):
        return imgs

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = torch.cat((x[0], x[1]), 3)
        x = np.swapaxes(x, 1, 3)
        x = np.swapaxes(x, 2, 3)

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv3_1 = self.conv3_1(out_conv3)
        out_conv4 = self.conv4(out_conv3_1)
        out_conv4_1 = self.conv4_1(out_conv4)
        out_conv5 = self.conv5(out_conv4_1)
        out_conv5_1 = self.conv5_1(out_conv5)
        out_conv6 = self.conv6(out_conv5_1)

        return 0

    def validation_step(self, val_batch, batch_idx):
        pass

    def configure_optimizers(self):
        # TODO: we need a learning rate scheduler.After the first 300k iteration,
        #  every 100k iterations the lr should be divided by 2.
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) # keep the betas and the eps as default
        return optimizer



