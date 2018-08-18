import numpy as np
import os
from importlib import import_module
import time

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import data_loader


n_workers=10  #data loader workers
#n_gpu=3
start_lr=0.01
weight_decay=1e-4
nb_epochs=30
save_freq=1

kernel_size=7
#data_dir='/home/wyc/Desktop/preprocessed_data'
data_dir='/home/wyc/Desktop/toy_dataset'
win_width=32  #label window, input window=win_width+kernel_size-1
batch_size=256  #256=32*8=256*1
model_path='conv'  
save_dir='/home/wyc/Desktop/model_save'  #model save
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

model = import_module(model_path)
net, loss= model.get_model()

net = net.cuda()
loss = loss.cuda()
cudnn.benchmark = True
net = DataParallel(net) 

dataset=data_loader.data_loader(data_dir,win_width, kernel_size,overlap=True,phase='train')
train_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = n_workers,
        pin_memory=True)   #train/val pin_memory=True, test pin_memory=False

dataset=data_loader.data_loader(data_dir,win_width, kernel_size,overlap=True,phase='val')
val_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_workers,
        pin_memory=True)   #train/val pin_memory=True, test pin_memory=False

# dataset=data_loader.data_loader(data_dir,win_width, kernel_size,overlap=True,phase='test')
# test_loader = DataLoader(
#         dataset,
#         batch_size = batch_size,
#         shuffle = False,
#         num_workers = n_workers,
#         pin_memory=True)   #train/val pin_memory=True, test pin_memory=False,not the real test

optimizer = optim.SGD(
        net.parameters(),
        start_lr,
        momentum = 0.9,
        weight_decay = weight_decay)


def get_lr(epoch,nb_epochs,start_lr):
    if epoch <= nb_epochs * 0.5:
        lr = start_lr
    elif epoch <= nb_epochs * 0.8:
        lr = 0.1 * start_lr
    else:
        lr = 0.01 * start_lr
    return lr


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,nb_epochs,start_lr):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch,nb_epochs,start_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target) in enumerate(data_loader):
        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))

        output = net(data)
        loss_output = loss(output,target)#(8L, 88L, 32L, 1L)/(8L, 1L, 32L, 88L)
        #print loss_output[0].shape
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)

    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict
            },
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)

    print('Epoch %03d (lr %.5f),time %3.2f' % (epoch, lr,end_time - start_time))  # Framewise and notewise Accuracy precision,recall,F-score
    TP=np.sum(metrics[:, 1])
    Precision=TP/np.sum(metrics[:, 2])
    Recall=TP/np.sum(metrics[:, 3])
    Fscore=2*Precision*Recall/(Precision+Recall)
    print('Train:　loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics[:,0]),Precision,Recall,Fscore))
    print


def validate(data_loader, net, loss):
    start_time = time.time()

    net.eval()

    metrics = []
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data.cuda(async=True), volatile=True)
        target = Variable(target.cuda(async=True), volatile=True)

        output = net(data)
        loss_output = loss(output, target, train=False)

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    TP=np.sum(metrics[:, 1])
    Precision=TP/np.sum(metrics[:, 2])
    Recall=TP/np.sum(metrics[:, 3])
    Fscore=2*Precision*Recall/(Precision+Recall)
    print('Validation: Loss %2.4f,Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics[:,0]),Precision,Recall,Fscore))
    print
    print



if __name__=='__main__':
    start_epoch=0
    for epoch in range(start_epoch,nb_epochs):
        train(train_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,nb_epochs,start_lr)
        validate(val_loader, net, loss)

