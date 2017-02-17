import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.c1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(channel)
        self.c2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.b2 = nn.BatchNorm2d(channel)
    def forward(self, x):
        hidden = F.relu(self.b1(self.c1(x)))
        res = x + self.b2( self.c2(hidden))
        return res

class Vgg16Part(nn.Module):
    def __init__(self):
        super(Vgg16Part, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #print('start forward')
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        f1 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        #print('gogo')
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        f2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        #print('gogo')
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        f3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        # print('gogo')
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        f4 = h
        # print('gogo')
        # [relu1_2,relu2_2,relu3_3,relu4_3]
        return [f1, f2, f3, f4]

class StylePart(nn.Module):
    def __init__(self):
        super(StylePart, self).__init__()
        self.c1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(64) 
        self.c3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.b3 = nn.BatchNorm2d(128)       
        self.r1 = ResBlock(128)
        self.r2 = ResBlock(128)
        self.r3 = ResBlock(128)
        self.r4 = ResBlock(128)
        self.r5 = ResBlock(128)     
        self.d1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.b4 = nn.BatchNorm2d(64)
        self.d2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.b5 = nn.BatchNorm2d(32)
        self.d3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = F.relu(self.b2(self.c2(h)))
        h = F.relu(self.b3(self.c3(h)))
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = self.r4(h)
        h = self.r5(h)
        h = F.relu(self.b4(self.d1(h)))
        h = F.relu(self.b5(self.d2(h)))
        y = self.d3(h)
        return y