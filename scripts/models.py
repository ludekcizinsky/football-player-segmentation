import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def CBR(in_feat, out_feat, kernel_size=3, stride=1, padding=1):
            """Convolution -> Batch normalization -> ReLU"""
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )
        
        self.down1 = CBR(in_channels, 64)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), CBR(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), CBR(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), CBR(256, 512))
        
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), CBR(512, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = CBR(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = CBR(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = CBR(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = CBR(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        b = self.bottleneck(d4)
        
        u1 = self.up1(b)
        u1 = torch.cat((u1, d4), dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat((u2, d3), dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat((u3, d2), dim=1)
        u3 = self.conv_up3(u3)
        
        u4 = self.up4(u3)
        u4 = torch.cat((u4, d1), dim=1)
        u4 = self.conv_up4(u4)
        
        out = self.final_conv(u4)
        return out
