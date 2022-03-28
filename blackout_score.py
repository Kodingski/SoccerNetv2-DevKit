# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:42:47 2022

@author: timsc
"""

#%%imports
import moviepy.editor as mp
import numpy as np
#%%load video

video = mp.VideoFileClip("Data/SF_02ST_SGE_FCA.mp4")
myclip = video.subclip(0,10)

#%%define filter
def blackout_filter(frame):
    frame_copy = np.copy(frame)
    frame_copy[66:112,1250:1850,:] *= 0
    return frame_copy

#%%blackout the score and save file
clip_with_blackouts = myclip.fl_image(blackout_filter)
clip_with_blackouts.write_videofile("Data/blacked_out.mp4")
