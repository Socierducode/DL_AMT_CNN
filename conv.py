#CNN model
import torch
from torch import nn
from layers import *

# config = {}
# config['anchors'] = [10.0, 30.0, 60.]
# config['chanel'] = 1
# config['crop_size'] = [128, 128, 128]
# config['stride'] = 4
# config['max_stride'] = 16
# config['num_neg'] = 800
# config['th_neg'] = 0.02
# config['th_pos_train'] = 0.5
# config['th_pos_val'] = 1
# config['num_hard'] = 2
# config['bound_size'] = 12
# config['reso'] = 1
# config['sizelim'] = 6.  # mm
# config['sizelim2'] = 30
# config['sizelim3'] = 40
# config['aug_scale'] = True
# config['r_rand_crop'] = 0.3
# config['pad_value'] = 170
# config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
# config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
#                        'adc3bbc63d40f8761c59be10f1e504c3']


# config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','d92998a73d4654a442e6d6ba15bbb827','990fbe3f0a1b53878669967b9afd1441','820245d8b211808bd18e78ff5be16fdb','adc3bbc63d40f8761c59be10f1e504c3',
#                       '417','077','188','876','057','087','130','468']

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
    loss = Loss()#config['num_hard'])
    return net, loss