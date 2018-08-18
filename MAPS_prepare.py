#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 20:51:08 2016

@author: wyc

Preprocessing module specific to the MAPS dataset
Multi-thread is supported
"""

import re,os
import os.path as osp
import glob
import numpy as np
import random
import librosa as lb
#import midi_to_mat as mm
import pretty_midi
import zipfile
import sys
from multiprocessing import Pool
from functools import partial

#import h5py
eps=sys.float_info.epsilon
pretty_midi.pretty_midi.MAX_TICK = 1e10
## Paramater setting ##
RangeMIDInotes=[21,108]
sr=44100.
bins_per_octave=36
n_octave=7
maps_path='/home/wyc/Desktop/MAPS'
test_list=['ENSTDkAm','ENSTDkCl']   #real piano
val_rate=1./7
n_workers=9
    
    
def MAPS_unzip(maps_path,piano_model_list=None,sub_block='MUS',temp_path=None,n_worker=n_workers):
#Generate a standarised training file from the MAP file

#sub_piano_model: piano model name(s) to choose,can be a str or a list, None means all .zip files
# Example: MAPS_ENSTDkAm_1.zip
# sub_piano_model='MAPS_ENSTDkAm_1': 1 file ouputs
# sub_piano_model='MAPS_ENSTDkAm': 2 files output

# sub_block = ISOL/RAND/UCHO/MUS (extract full pieces of music by default)
# output: Path to store the selected files 

    file_list_full=glob.glob(osp.join(maps_path,'*'+'.zip'))
    file_list=[]

# define piano model list
    if piano_model_list == None:
        file_list=file_list_full   #full songs
        piano_model_list=[]
        for i in file_list:
            piano_model_list.append(i.split('_')[1])
        piano_model_list=set(piano_model_list)

    if type(piano_model_list)==str:
        piano_model_list=[piano_model_list]

    for i in range(len(file_list_full)):
        for j in piano_model_list:
            if re.search(j,osp.basename(file_list_full[i]))!=None:
                file_list.append(file_list_full[i])

    if file_list==[]:
        Exception('A bad piano model list inserted')

# create temp_path dic if not exist
    if temp_path==None:
        temp_path=osp.join(osp.dirname(osp.realpath(maps_path)),'selected_MAPS_files/')
    if not osp.exists(temp_path):
        os.makedirs(temp_path)

    for i in piano_model_list:
        if not osp.exists(temp_path+i):
            os.makedirs(temp_path+i)

    if type(sub_block)==str:
        sub_block=[sub_block]

#Decompress the selected files          #unziping the whole dataset takes 8 mins on i7-4720 CPU 33% occupied single thread
    if n_workers==1 or len(piano_model_list)==1:
        for k in file_list:
            unzip(k, sub_block, temp_path)
    else:
        pool = Pool(processes=n_worker)
        partial_processing = partial(unzip, sub_block=sub_block,)
        _ = pool.map(partial_processing, file_list)
        pool.close()
        pool.join()
    print "Unzip file succeed!"  
    return temp_path

def unzip(k,sub_block,temp_path):
    f = zipfile.ZipFile(k, 'r')
    for eachfile in f.namelist():
        for b in sub_block:
            if re.search(b, eachfile) != None:
                eachfilename = osp.normpath(osp.join(temp_path, eachfile.split('/')[0] + '/' + eachfile.split('/')[2]))
                if eachfile[-3:] in ['mid', 'wav']:  # extract only mid and wav files
                    print "Write unzipped file %s ..." % eachfilename
                    fd = open(eachfilename, "wb")
                    fd.write(f.read(eachfile))
                    fd.close()
    f.close()

def train_val_test_prepa(file_path,output=None,test_list=test_list,val_rate=val_rate) :
# file_path: Path of the sources, ex. unzipped MAPS subblock     
# output: Path to store the train/test files 
# test_rate: randomly choose the test files with test_rate

# delect: empty the existed selected_MAPS_files if delect=1

# Directories creation
    if output==None:
        output=osp.join(osp.dirname(osp.realpath(file_path)),'data')
        if not osp.exists(output):
            os.makedirs(output)
    output_train=osp.join(output,'train')
    output_val=osp.join(output,'val')
    output_test=osp.join(output,'test')    
    
    if not osp.exists(output_train):
        os.makedirs(output_train)

    if not osp.exists(output_val):
        os.makedirs(output_val)

    if not osp.exists(output_test):
        os.makedirs(output_test)

# Move the test set
    if type(test_list)==str:
        test_list=[test_list]

    for i in test_list:
        test_path=glob.glob(file_path+i+'/*')
        for j in test_path:
            print "Write val wav/mid files: %s ..." %j
            os.system("cp %s %s" % (j,output_test))
        os.system("rm -r %s" % (file_path+i))

# Read all files
    file_name=[]
    for root,dirs,files in os.walk(file_path):  
        for file in files:
            a=os.path.join(root,file)
            if a[-3:] in ['mid','wav']: 
               file_name.append(a[:-3])    
    file_name=list(set(file_name))  #Discard repeated names
    
# Randomly choose the test files

    random.shuffle(file_name)
    file_val=file_name[:int(len(file_name)*val_rate)]
    file_train=file_name[int(len(file_name)*val_rate):]

    if type(file_val)==str:
        file_test=[file_val]
    if type(file_train) == str:
        file_train = [file_train]

# Write into the test file
    for i in file_val:
        print "Write val wav/mid files: %s ..." % i
        os.system("cp %s %s" % (i+'wav',osp.normpath(osp.join(output_val, osp.basename(i)+'wav'))))
        os.system("cp %s %s" % (i + 'mid', osp.normpath(osp.join(output_val, osp.basename(i) + 'mid'))))

# Write the rest files into train file
    for i in file_train:
        print "Write train wav/mid files: %s ..." % i
        os.system("cp %s %s" % (i+'wav',osp.normpath(osp.join(output_train, osp.basename(i)+'wav'))))
        os.system("cp %s %s" % (i + 'mid', osp.normpath(osp.join(output_train, osp.basename(i) + 'mid'))))
    print "Train/val/test files created!"
    os.system("rm -r %s" % (temp_path))
    return output


def preprocessing(data_path, sr=sr, bins_per_octave=bins_per_octave, n_octave=n_octave,#win_width=3,
                  RangeMIDInotes=RangeMIDInotes, save_path=None,n_worker=n_workers,delete=True):
    # Convert the raw data(wav/mid) into input/output data from the train/test directories

    # Net architecture
    # conv_mode=　context based
    # framewise: convolution works only on a single frame,along the frequency axis
    # multi-frame: convolution works on several frames

    # win-width: number of the frames when conv_mode=multiframe or multi-frame

    # data_path = None or any other with train/test dirs inside

    # output_path: Path to save the processed data with format hdf5
    # None=only preprocessed data,no output file;
    # '' generate an output directory in current directory(without preprocessed data)

    # output_name: name the hf file, data.h5 by default
    # sr:Raw audio sampling rate
    # RangeMIDInotes: by default for the 88 key piano

    # Default data path
    if save_path == None:
        save_path = osp.join(osp.dirname(osp.realpath(data_path)), 'preprocessed_data')
        if not osp.exists(save_path):
            os.makedirs(save_path)
    output_train = osp.join(save_path, 'train')
    output_val = osp.join(save_path, 'val')
    output_test = osp.join(save_path, 'test')

    if not osp.exists(output_train):
        os.makedirs(output_train)

    if not osp.exists(output_val):
        os.makedirs(output_val)

    if not osp.exists(output_test):
        os.makedirs(output_test)

    # train/test inside
    train_list = glob.glob(osp.join(data_path, 'train') + '/*')
    val_list = glob.glob(osp.join(data_path, 'val') + '/*')
    test_list = glob.glob(osp.join(data_path, 'test') + '/*')

    train_name = []
    val_name= []
    test_name = []
    for i in train_list:
        train_name.append(i[:-3])
    for i in val_list:
        val_name.append(i[:-3])
    for i in test_list:
        test_name.append(i[:-3])
    train_name = list(set(train_name))
    val_name = list(set(val_name))
    test_name = list(set(test_name))

    n_bins=n_octave*bins_per_octave

    # Set creation
    # Initialization
    # if win_width == 1:
    #     X_train = []
    #     X_val=[]
    #     X_test = []
    # else:
    #     zero_pad = int(win_width / 2)  # window width had better being a odd number
    #     X_train = np.zeros([1, n_bins, win_width])
    #     X_test = np.zeros([1, n_bins, win_width])
    # Y_train = np.zeros([1, RangeMIDInotes[1] - RangeMIDInotes[0] + 1])
    # Y_test = np.zeros([1, RangeMIDInotes[1] - RangeMIDInotes[0] + 1])

    # training set processing
    if n_worker==1:
        for i in train_name:
            processing(i, n_bins, output_train, sr=sr, bins_per_octave=bins_per_octave,
                    RangeMIDInotes=RangeMIDInotes)
        for i in val_name:
            processing(i, n_bins, output_val, sr=sr, bins_per_octave=bins_per_octave,
                    RangeMIDInotes=RangeMIDInotes)
    # testing set processing# testing set processing
        for i in test_name:
            processing(i, n_bins, output_test, sr=sr, bins_per_octave=bins_per_octave,
                    RangeMIDInotes=RangeMIDInotes)
    else:
        pool = Pool(processes=n_worker)
        partial_processing = partial(processing, n_bins=n_bins, output=output_train, sr=sr, bins_per_octave=bins_per_octave,
                    RangeMIDInotes=RangeMIDInotes)
        _ = pool.map(partial_processing, train_name)
        pool.close()
        pool.join()

        pool = Pool(processes=n_worker)
        partial_processing = partial(processing, n_bins=n_bins,output=output_val, sr=sr,
                                     bins_per_octave=bins_per_octave,
                                     RangeMIDInotes=RangeMIDInotes)
        _ = pool.map(partial_processing, val_name)
        pool.close()
        pool.join()

        pool = Pool(processes=n_worker)
        partial_processing = partial(processing, n_bins=n_bins, output=output_test, sr=sr,
                                     bins_per_octave=bins_per_octave,
                                     RangeMIDInotes=RangeMIDInotes)
        _ = pool.map(partial_processing, test_name)
        pool.close()
        pool.join()
    print 'Data preprocessing completed'
    if delete:
        os.system("rm -r %s" % (data_path))

def processing(data_path,n_bins,output,sr=sr, bins_per_octave=bins_per_octave,
                  RangeMIDInotes=RangeMIDInotes):
    save_path=osp.join(output,data_path.split('/')[-1][:-1])
    # input:  CQT spectrum form raw audio
    audio_path_train = data_path + 'wav'
    x, sr = lb.load(audio_path_train, sr=sr)
    CQT_spectrum = lb.cqt(x, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins,
                                fmin=lb.note_to_hz('A0'), real=False)
    CQT = np.transpose(np.abs(CQT_spectrum))

    # Ground-truth: convert midi to pianoroll
    midi_path_train = data_path + 'mid'
    Ground_truth_mat=midi2mat(midi_path_train, len(x), CQT.shape[0], sr, RangeMIDInotes=RangeMIDInotes)
    midi_train = np.transpose(Ground_truth_mat)

    if midi_train.shape[0]<CQT.shape[0]:
    #midi length<CQT length, cut CQT
        CQT=CQT[:midi_train.shape[0],:]
    np.save(save_path + '_CQT.npy', CQT)
    np.save(save_path + '_label.npy', midi_train)
    print "Preprocessing of file %s completed..." % (data_path[:-1])

def midi2mat(midi_path_train, length, CQT_len, sr, RangeMIDInotes=RangeMIDInotes):
    midi_data = pretty_midi.PrettyMIDI(midi_path_train)
    pianoRoll = midi_data.instruments[0].get_piano_roll(fs=CQT_len * sr/length)
    Ground_truth_mat = (pianoRoll[RangeMIDInotes[0]:RangeMIDInotes[1] + 1, :CQT_len] > 0)
    return Ground_truth_mat

if __name__=='__main__':
    # 10 mins
    temp_path=MAPS_unzip(maps_path,sub_block='MUS',n_worker=n_workers)
    # moving the file takes about 3 mins, note: the file with the temp_path will be deleted after executing train_val_test_prepa
    output_path=train_val_test_prepa(temp_path,test_list=test_list,val_rate=val_rate)
    #output_path = '/home/wyc/Desktop/data'
    preprocessing(output_path, sr=sr, bins_per_octave=bins_per_octave, n_octave=n_octave,
                    RangeMIDInotes=RangeMIDInotes,n_worker=n_workers,delete=False)
        
        
        

