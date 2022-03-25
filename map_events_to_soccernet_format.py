# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:02:03 2022

@author: timsc
"""

import json
import moviepy.editor as mp
import xml.etree.ElementTree as ET

from xml.dom import minidom

#%%load
# parse an xml file by name
tree = ET.parse('Data/DFL-MAT-003CPF_full_video_labels_uniform_setpieces_2.5_0.4_1.0_214.5_4074.45.xml')

##to do: find way to organise matches in a list like in their format
##       For now focus on one match and reproduce their annotations


#%%make dictionary to later transform into .json
events_dict = {}
events_dict["annotations"] = []

#%%
##find all events (instances) in the xml
lst = tree.findall('ALL_INSTANCES/instance')
print('Event Count:', len(lst))

#%%
#counter to keep track of event time in trimmed video
event_counter = 0
#match_id = 


#add info about event in dictionaries similar to the style of NetVlad ++ 
for event in lst:
    
    starting_time = event_counter*2.5
    
    event_dict = {}

    event_dict["start_raw"] = event.find('start').text
    event_dict["end_raw"] = event.find('end').text
    
    event_dict["id"] = event_counter
    event_dict["start"] = starting_time
    event_dict["end"] = starting_time + 2.5
    
    

    #initialiaze timesteps of actual events 
    event_dict['timestep_raw'] = None
    event_dict['timestep'] = None

    
    #if an event exists in clip (and therefore also a timestamp), write in dict
    labels = event.findall("label")
    for label in labels:
        if label.find("group").text == 'Event_Timestamp':
            print(label.find('text').text)
            event_dict['timestep_raw'] = label.find('text').text
            event_dict['timestep'] = starting_time + (float(event_dict['timestep_raw']) - float(event_dict["start_raw"])) 
    
            


    
    event_dict["type"] = event.find("code").text
    

    
    
    
    event_counter += 1
    events_dict["annotations"].append(event_dict)
    
