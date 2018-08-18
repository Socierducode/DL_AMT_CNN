import numpy as np

import torch
from torch import nn
import math


def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


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



def topkpbb(pbb, lbb, nms_th, detect_th, topk=30):
    conf_th = 0
    fp = []
    tp = []
    while len(tp) + len(fp) < topk:
        conf_th = conf_th - 0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th < -3:
            break
    tp = np.array(tp).reshape([len(tp), 6])
    fp = np.array(fp).reshape([len(fp), 6])
    fn = np.array(fn).reshape([len(fn), 5])
    allp = np.concatenate([tp, fp], 0)
    sorting = np.argsort(allp[:, 0])[::-1]
    n_tp = len(tp)
    topk = np.min([topk, len(allp)])
    tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
    fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
    #     print(fp_in_topk)
    fn_i = np.array([i for i in range(n_tp) if i not in sorting[:topk]])
    newallp = allp[:topk]
    if len(fn_i) > 0:
        fn = np.concatenate([fn, tp[fn_i, :5]])
    else:
        fn = fn
    if len(tp_in_topk) > 0:
        tp = tp[tp_in_topk]
    else:
        tp = []
    if len(fp_in_topk) > 0:
        fp = newallp[fp_in_topk]
    else:
        fp = []
    return tp, fp, fn