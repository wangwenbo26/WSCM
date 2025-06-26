import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)
        self.conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(1, 1)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)

    def forward(self, x):
        x = self.upconv4(x)
        x = self.residual4(x)
        x = self.upconv3(x)
        x = self.residual3(x)
        x = self.upconv2(x)
        x = self.residual2(x)
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.conv(x)
        return x
