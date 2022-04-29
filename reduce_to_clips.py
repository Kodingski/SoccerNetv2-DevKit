# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:29:07 2022

@author: timsc
"""



#%% imports

import json
import moviepy.editor as mp
import xml.etree.ElementTree as ET

from xml.dom import minidom

#%%load final snippets of all games
# parse an xml file by name





tree = ET.parse('Data/DFL-MAT-003CPF_full_video_labels_uniform_setpieces_2.5_0.4_1.0_214.5_4074.45.xml')

##to do: find way to organise matches in a list like in their format
##       For now focus on one match and reproduce their annotations


#%%make dictionary to later transform into .json
events_dict = {}
events_dict["annotations"] = []

#%%load video data and initiate clip list
video = mp.VideoFileClip("Data/SF_02ST_SGE_FCA.mp4")
video.resize( (398,224) )
clips = []

#%%
##find all events (instances) in the xml
lst = tree.findall('ALL_INSTANCES/instance')
print('Event Count:', len(lst))

#counter to keep track of event time in trimmed video
event_counter = 0
#match_id = 

#add info about event in dictionaries similar to the style of NetVlad ++ 
for event in lst:
    
    if event_counter % 100 == 0:
        print(event_counter)
    event_dict = {}

    event_dict["start_raw"] = event.find('start').text
    event_dict["end_raw"] = event.find('end').text
    event_dict["type"] = event.find("code").text
    
    event_dict["id"] = event_counter
    event_dict["start"] = event_counter * 2.5
    event_dict["end"] = event_counter * 2.5 + 2.5
    
    event_counter += 1
    events_dict["annotations"].append(event_dict)
    
    clip = video.subclip(float(event_dict["start_raw"]), float(event_dict["end_raw"]))
    clips.append(clip)
    
final_clip = mp.concatenate_videoclips(clips)



#%%
final_clip.write_videofile(filename = "Data/SGE_FCA_snippets.mp4", 
                               audio = False, threads = 8)




