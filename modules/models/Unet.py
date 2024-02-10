import torch.nn as nn

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.mp = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x, downsample=True):

        if downsample:
          x = self.mp(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn0 = nn.BatchNorm2d(in_channels//2)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels//2)
        self.relu2 = nn.ReLU()

    def forward(self, x1, x2):

        x1 = self.relu0(self.bn0(self.upconv(x1)))

        x = torch.cat([x2, x1], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class UNet(nn.Module):
  def __init__(self, num_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.ds1 = DownSamplingBlock(64, 64)
        self.ds2 = DownSamplingBlock(64, 128)
        self.ds3 = DownSamplingBlock(128, 256)
        self.ds4 = DownSamplingBlock(256, 512)
        self.ds5 = DownSamplingBlock(512, 1024)

        self.us1 = UpSamplingBlock(1024)
        self.us2 = UpSamplingBlock(512)
        self.us3 = UpSamplingBlock(256)
        self.us4 = UpSamplingBlock(128)

        self.conv2 = nn.Conv2d(64, 1, 1)



  def forward(self, x):

      x0 = x
      x1 = self.relu1(self.bn1(self.conv1(x0)))
      
      x2 = self.ds1(x1, False)
      x3 = self.ds2(x2)
      x4 = self.ds3(x3)
      x5 = self.ds4(x4)
      x6 = self.ds5(x5)

      x7 = self.us1(x6, x5)
      x8 = self.us2(x7, x4)
      x9 = self.us3(x8, x3)
      x10 = self.us4(x9, x2)

      x11 = self.conv2(x10)
      x = x0 + x11
      return x
      