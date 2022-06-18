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

class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()


        self.enc1 = DoubleConv_3d(1, 16)
        self.enc2 = DoubleConv_3d(16, 32)
        self.enc3 = DoubleConv_3d(32, 64)
        self.enc4 = DoubleConv_3d(64, 128)
        self.enc5 = DoubleConv_3d(128, 128)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.dec4 = DoubleConv_3d(256, 64)
        self.dec3 = DoubleConv_3d(128, 32)
        self.dec2 = DoubleConv_3d(64, 16)
        self.dec1 = DoubleConv_3d(32, 16)

        self.conv_final = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        dec4 = self.dec4(torch.cat((enc4, F.interpolate(enc5, enc4.shape[2:], mode='trilinear')), 1))
        dec3 = self.dec3(torch.cat((enc3, F.interpolate(dec4, enc3.shape[2:], mode='trilinear')), 1))
        dec2 = self.dec2(torch.cat((enc2, F.interpolate(dec3, enc2.shape[2:], mode='trilinear')), 1))
        dec1 = self.dec1(torch.cat((enc1, F.interpolate(dec2, enc1.shape[2:], mode='trilinear')), 1))
        out = self.conv_final(dec1)
        out = torch.sigmoid(out)
        return out