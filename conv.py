#CNN model
import torch
from torch import nn
from layers import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), return_indices=False)

        self.conv1= nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(5,25), stride=1),
            nn.BatchNorm2d(50),
            nn.ELU(inplace=True))
            #nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(3,5), stride=1),
            nn.BatchNorm2d(50),
            nn.ELU(inplace=True))
        # nn.ReLU(inplace=True)
        self.conv3=nn.Sequential(
            nn.Conv2d(50, 1000, kernel_size=(1,24)),   #kernel_size !!!!! conv_mode,padding=0 by default
            #nn.ReLU(inplace=True),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.conv4 = nn.Sequential(
            nn.Conv2d(1000, 500, kernel_size=1),  # kernel_size !!!!! conv_mode,padding=0 by default
            nn.ELU(inplace=True),
        #nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.conv5=nn.Sequential(
            nn.Conv2d(500, 88, kernel_size=1))#,
            #nn.Sigmoid())

    def forward(self, x):
        # (8L, 1L, 38L, 252L)
        x = self.conv1(x)#(8L, 50L, 34L, 228L)
        x = self.maxpool(x)#(8L, 50L, 34L, 76L)
        x = self.conv2(x)  #(8L, 50L, 32L, 72L)
        x = self.maxpool(x)#(8L, 50L, 32L, 24L)
        x = self.conv3(x)# (8L, 1000L, 32L, 1L)
        x = self.conv4(x)#(8L, 500L, 32L, 1L)
        x = self.conv5(x)#(8L, 88L, 32L, 1L)
        #print x.shape
        return x


def get_model():
    net = Net()
    loss = Loss()
    return net, loss
