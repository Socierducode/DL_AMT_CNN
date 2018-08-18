import numpy as np

import torch
from torch import nn
import math


class Loss(nn.Module):
    def __init__(self, num_hard=0):
        super(Loss, self).__init__()
        #self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCEWithLogitsLoss() #weight is defined for samples
        #self.num_hard = num_hard

    def forward(self, output, labels, train=True):
        #batch_size = labels.size(0)
        #output = torch.transpose(output,0,3,2,1) #[6,88,32,1]->[4224,4]
        #output=output.view(-1,88)
        #labels = labels.view(-1,88)#[6,1,32,88]->[4224,4] unbalance
        #print output.shape,labels.shape

        loss = self.classify_loss(
           output,labels)
        pos = (torch.sigmoid(output) >= 0.5).type(torch.cuda.FloatTensor)
        pos_recall=labels.sum()
        pos_precision=pos.sum()
        TP=(pos*labels).sum()
        #print pos_recall.shape
        #print type(pos)
        return [loss,TP.data[0], pos_precision.data[0], pos_recall.data[0]] #F-score must be computed by whole epoch
