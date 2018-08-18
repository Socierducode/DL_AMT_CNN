import numpy as np
#import random
import os,glob
from torch.utils.data import Dataset
#import time
import torch

#from multiprocessing import Pool
#from functools import partial

# win_width=100
# kernel_size=7  #7*252=42**2=1764
# data_dir='/home/wyc/Desktop/preprocessed_data'


class data_loader(Dataset):
    def __init__(self, data_dir, win_width, kernel_size,overlap=True,phase='train'):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        filelist= glob.glob(os.path.join(data_dir,phase)+'/*')
        CQT_name=[f for f in filelist if (f[-7:-4]=='CQT')]
        self.input=[]
        self.nb_sample=[]

        for i in CQT_name:
            self.input.append(np.expand_dims(cut(np.load(i),win_width,kernel_size,overlap=overlap),axis=1))   # 64s,no need to paralellize, I/O is limited ,246s with 5 workers
            self.nb_sample.append(self.input[-1].shape[0])
        #input=np.concatenate(input,axis=0) #don't use the concatenate,very time-consuming (35978, 1, 106, 252)

        #if self.phase != 'test':   #Attention: we have the ground truth for testset
        label_name=[f[:-7]+'label.npy' for f in CQT_name]
        self.label=[]
        for i in label_name:
            self.label.append(np.expand_dims(cut(np.load(i),win_width,kernel_size,overlap=False).transpose(0,2,1),axis=3))

    def __getitem__(self,idx):
        #if self.phase!='test':  no real test in our senerios
        nb_list, sub_nb = index(idx,self.nb_sample)
        return torch.from_numpy(self.input[nb_list][sub_nb].astype(np.float32)),torch.from_numpy(self.label[nb_list][sub_nb].astype(np.float32))   #(1,106,252)/(1,100,88)

        # else:    no real test in our senerios
        #     nb_list, sub_nb = index(idx, self.nb_sample)
        #     return torch.from_numpy(self.input[nb_list][sub_nb])
    def __len__(self):
        return sum(self.nb_sample)

def cut(matrix,win_width,kernel_size,overlap=True,axis=0):  #window cut module
# cut the tensor along the first axis by the win_width with a single frame hop
    #matrix=np.load(matrix)
    l=matrix.shape[0]
    cut_matrix=[]
    nb_win=l/win_width   #integer division=floor
    if not overlap:
        for i in xrange(nb_win):
            cut_matrix.append(matrix[i*win_width:(i+1)*win_width,:])
    else:
        w=matrix.shape[1]
        matrix_1=np.concatenate([np.zeros([kernel_size/2,w]),matrix,np.zeros([kernel_size/2,w])],axis=0)  #padding
        cut_matrix = []
        for i in xrange(nb_win):
            cut_matrix.append(matrix_1[i * win_width:(i + 1) * win_width+kernel_size-1,:])    #0-104,100-204,...
    cut_matrix = np.asarray(cut_matrix)
    return cut_matrix

def index(idx,nb_sample):
    l=len(nb_sample)
    accum_nb =0
    nb_list=0
    sub_nb=0
    for i in range(l):
        accum_nb+=nb_sample[i]
        if idx < accum_nb:
            nb_list, sub_nb= i, idx+nb_sample[i]-accum_nb
            break
    return nb_list,sub_nb