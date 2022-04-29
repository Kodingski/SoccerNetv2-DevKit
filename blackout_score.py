# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:42:47 2022

@author: timsc
"""
#%% imports

import json
import moviepy.editor as mp
import numpy as np
import xml.etree.ElementTree as ET

from xml.dom import minidom


#%%load video

video = mp.VideoFileClip(r"D:/data/SF_02ST_BOC_M05.mp4")
myclip = video.subclip(6360,6420)

#%%define filter
def blackout_filter(frame):
    frame_copy = np.copy(frame)
    frame_copy[0:215,0:1920,:] *= 0
    return frame_copy

#%%blackout the score and save file
clip_with_blackouts = myclip.fl_image(blackout_filter)
clip_with_blackouts.write_videofile(r"D:/data/SF_02ST_BOC_M05_blacked_out.mp4")
