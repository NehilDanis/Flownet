import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np


def conv(in_channel, out_channel, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=padding),
        nn.ReLU()
    )

def transposedConv(in_channel, out_channel):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False),
        nn.ReLU()
    )


def crop_like(input, target):
    if input.shape[2: ] == target.shape[2: ]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def mean_EPE(input_flow, target_flow):
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

        self.deconv5 = transposedConv(1024, 512)
        self.deconv4 = transposedConv(1024, 256)
        self.deconv3 = transposedConv(770, 128)
        self.deconv2 = transposedConv(386, 64)


        self.predict_flow5 = predict_flow(1024)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 5, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 5, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 5, 2, 1, bias=False)

        self.upsample_bilinear = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv3_1 = self.conv3_1(out_conv3)
        out_conv4 = self.conv4(out_conv3_1)
        out_conv4_1 = self.conv4_1(out_conv4)
        out_conv5 = self.conv5(out_conv4_1)
        out_conv5_1 = self.conv5_1(out_conv5)
        out_conv6 = self.conv6(out_conv5_1)

        out_deconv5 = self.deconv5(out_conv6)
        input_to_deconv4 = torch.cat((crop_like(out_deconv5, out_conv5_1), out_conv5_1), 1)
        flow5 = self.predict_flow5(input_to_deconv4)

        upsampled_flow5_to_4 = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4_1)
        out_deconv4 = self.deconv4(input_to_deconv4)
        input_to_deconv3 = torch.cat((crop_like(out_deconv4, out_conv4_1), out_conv4_1, upsampled_flow5_to_4), 1)
        flow4 = self.predict_flow4(input_to_deconv3)

        upsampled_flow4_to_3 = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3_1)
        out_deconv3 = self.deconv3(input_to_deconv3)
        input_to_deconv2 = torch.cat((crop_like(out_deconv3, out_conv3_1), out_conv3_1, upsampled_flow4_to_3), 1)
        flow3 = self.predict_flow3(input_to_deconv2)

        upsampled_flow3_to_2 = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = self.deconv2(input_to_deconv2)
        input_to_upsamling = torch.cat((crop_like(out_deconv2, out_conv2), out_conv2, upsampled_flow3_to_2), 1)

        flow2 = self.predict_flow2(input_to_upsamling)
        output = self.upsample_bilinear(flow2)

        return output

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = torch.cat((x[0], x[1]), 3)
        x = np.swapaxes(x, 1, 3)
        x = np.swapaxes(x, 2, 3)
        y = np.swapaxes(y, 1, 3)
        y = np.swapaxes(y, 2, 3)
        prediction = self.forward(x)
        loss = mean_EPE(prediction, y)

        print(f"Batch id:  {batch_idx} -----> Train Loss: {loss}")

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = torch.cat((x[0], x[1]), 3)
        x = np.swapaxes(x, 1, 3)
        x = np.swapaxes(x, 2, 3)
        y = np.swapaxes(y, 1, 3)
        y = np.swapaxes(y, 2, 3)
        prediction = self.forward(x)
        loss = mean_EPE(prediction, y)
        self.log('val_loss', loss)
        print(f"Batch id:  {batch_idx} -----> Validation Loss: {loss}")


    def configure_optimizers(self):
        # TODO: we need a learning rate scheduler.After the first 300k iteration,
        #  every 100k iterations the lr should be divided by 2.
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) # keep the betas and the eps as default
        return optimizer



