# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:29:07 2022

@author: timsc
"""



#%% imports

import json
import moviepy.editor as mp
import numpy as np
import xml.etree.ElementTree as ET

from xml.dom import minidom

#%%load final snippets of all games
# parse an xml file by name

#%%define filter
def blackout_filter(frame):
    frame_copy = np.copy(frame)
    frame_copy[66:112,1250:1850,:] *= 0
    return frame_copy

def reduce_to_clips(video, tree):
    
    event_counter = 0
    #make dictionary to later transform into .json
    events_dict = {}
    events_dict["annotations"] = []
    
    #load video data and initiate clip list

    video.resize( (398,224) )
    clips = []
    
    #find all events (instances) in the xml
    lst = tree.findall('ALL_INSTANCES/instance')
    
    #add info about event in dictionaries similar to the style of NetVlad ++ 
    for event in lst:
        #if event_counter == 100:
        #    break
        #print(event_counter)
        
        event_dict = {}
        event_dict["start_raw"] = event.find('start').text
        event_dict["end_raw"] = event.find('end').text

        clip = video.subclip(float(event_dict["start_raw"]), float(event_dict["end_raw"]))
        clips.append(clip)
        event_counter += 1 
    final_clip = mp.concatenate_videoclips(clips)

    return final_clip 

#%%
tree = ET.parse(r"C:\Users\timsc\OneDrive\Dokumente\kaggle-competition-bitbucket\data\final-snippets\xml\DFL-MAT-003CPF_214.5_4074.45_final_snippets.xml")
video = mp.VideoFileClip(r"D:/Data/SF_02ST_BSC_WOB.mp4")

#reduce to snippets
final_clip = reduce_to_clips(video, tree)

#blackout score
final_clip = final_clip.fl_image(blackout_filter)

print('here')  
#save
final_clip.write_videofile(filename = "D:/data/SF_02ST_BSC_WOB_snippets.mp4", audio = False, threads = 8)




