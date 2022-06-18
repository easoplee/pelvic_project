#3D U-Net model for multi-class segmentation
#TODO: figure out if activation function is needed at the end & what to pass in to num_channels

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv_3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_3d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.conv(x)
        return out

class UpConv_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2):
        super().__init__(UpConv_3d, self)

        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding,output_padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.upconv(x)
        return out

class UNet_3d(nn.Module):
    def __init__(self, num_channels):
        super(UNet_3d, self).__init__()

        #encoder
        self.down1 = DoubleConv_3d(in_channels=num_channels, out_channels=64)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down2 = DoubleConv_3d(in_channels=64, out_channels=128)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down3 = DoubleConv_3d(in_channels=128, out_channels=256)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down4 = DoubleConv_3d(in_channels=256, out_channels=512)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down5 = DoubleConv_3d(in_channels=512, out_channels=1024)

        #decoder
        self.upconv4 = UpConv_3d(in_channels=1024, out_channels=512)
        self.up4 = DoubleConv_3d(in_channels=1024, out_channels=512)
        self.upconv3 = UpConv_3d(in_channels=512, out_channels=256)
        self.up3 = DoubleConv_3d(in_channels=512, out_channels=256)
        self.upconv2 = UpConv_3d(in_channels=256, out_channels=128)
        self.up2 = DoubleConv_3d(in_channels=256, out_channels=128)
        self.upconv1 = UpConv_3d(in_channels=128, out_channels=64)
        self.up1 = DoubleConv_3d(in_channels=128, out_channels=64)

        #final 1x1 conv
        self.final_layer = nn.Conv3d(in_channels=64, out_channels=num_channels, kernel_size=1, stride=1, padding=0, biase=True)

    def forward(self, x):
        #encoder
        down1 = self.down1(x)
        skip1 = self.maxpool1(down1)
        down2 = self.down2(skip1)
        skip2 = self.maxpool2(down2)
        down3 = self.down3(skip2)
        skip3 = self.maxpool3(down3)
        down4 = self.down4(skip3)
        skip4 = self.maxpool4(down4)
        down5 = self.down5(skip4)

        #decoder with skip connections
        out = torch.cat((self.upconv4(down5), skip4), dim=1)
        out = self.up4(out)
        out = torch.cat((self.upconv3(out), skip3), dim=1)
        out = self.up3(out)
        out = torch.cat((self.upconv2(out), skip2), dim=1)
        out = self.up2(out)
        out = torch.cat((self.upconv1(out), skip1), dim=1)
        out = self.up1(out)

        #final layer
        out = self.final_layer(out)

        return out