# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:23:08 2022

@author: timsc
"""
#%%
from torch.utils.data import Dataset

import numpy as np
import random
import os
import time


from tqdm import tqdm

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1



#%%helpers

def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
        print(idx)
    # print(idx)
    return feats[idx,...]



#%% set to Sportec parameters
window_size = 2.5
framerate = 2
window_size_frame = int(window_size*framerate)
num_classes = 4


#%%
feat_game = np.load('Data/features_snippets.npy')
feat_game = feat_game.reshape(-1, feat_game.shape[-1])

#%%load their features to compare
feat_soccernet = np.load(r'england_epl\2014-2015\2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal\1_ResNET_TF2.npy')

window_size_soccernet = 15
framerate_soccernet = 2
window_size_frame_soccernet = window_size_soccernet*framerate_soccernet


#%%
feat_soccernet_head = feat_soccernet[0:100,:]
#%%
#feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
#feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

feat_game_sts = feats2clip(torch.from_numpy(feat_game), stride=window_size_frame, clip_length=window_size_frame)
#feat_soccernet_torch = feats2clip(torch.from_numpy(feat_soccernet), stride=window_size_frame_soccernet, clip_length=window_size_frame_soccernet)


#feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)


#%%
# Load soccernet labels
labels_soccernet = json.load(open(r'england_epl\2014-2015\2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal\Labels-v2.json'))

#%%recreate soccernet labels 
version = 2
dict_event = EVENT_DICTIONARY_V2
game_labels = []
num_classes = len(dict_event)

label_half1 = np.zeros((feat_game.shape[0], num_classes+1))
label_half1[:,0]=1 # those are BG classes
label_half2 = np.zeros((feat_game.shape[0], num_classes+1))
label_half2[:,0]=1 # those are BG classes


for annotation in labels_soccernet["annotations"]:

    time = annotation["gameTime"]
    event = annotation["label"]

    half = int(time[0])

    minutes = int(time[-5:-3])
    seconds = int(time[-2::])
    frame = framerate * ( seconds + 60 * minutes ) 

    if version == 1:
        if "card" in event: label = 0
        elif "subs" in event: label = 1
        elif "soccer" in event: label = 2
        else: continue
    elif version == 2:
        if event not in dict_event:
            continue
        label = dict_event[event]

    # if label outside temporal of view
    if half == 1 and frame//window_size_frame>=label_half1.shape[0]:
        continue
    if half == 2 and frame//window_size_frame>=label_half2.shape[0]:
        continue

    if half == 1:
        label_half1[frame//window_size_frame][0] = 0 # not BG anymore
        label_half1[frame//window_size_frame][label+1] = 1 # that's my class

    if half == 2:
        label_half2[frame//window_size_frame][0] = 0 # not BG anymore
        label_half2[frame//window_size_frame][label+1] = 1 # that's my class

#self.game_feats.append(feat_half2)
game_labels.append(label_half1)
game_labels.append(label_half2)
#self.game_labels.append(label_half2)

game_labels = np.concatenate(game_labels)

#%%load sportec lables
labels_sts = json.load(open(r'Data/SGE_FCA_annotations_snippets.json'))


##next: Recreate their one hot encoding on sportec data
dict_event_sts = {'None':0, 'Play':1, 'TacklingGame':2, 'Throw-in':3}

num_classes = len(dict_event_sts)
game_labels = np.zeros((len(labels_sts['annotations']), num_classes))


for annotation in labels_sts["annotations"]:

    snippet_id = annotation['id']
    end_time = annotation["end"]
    event_timestamp = annotation["timestep"]
    frame = framerate * ( seconds + 60 * minutes ) 
    
    event_type = annotation['type']
    label = dict_event_sts[event_type]

    game_labels[snippet_id,label] = 1 # that's my class


#%%do it with sportec labels