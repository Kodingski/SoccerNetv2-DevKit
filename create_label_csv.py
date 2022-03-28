# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:37:46 2022

@author: timsc
"""


#%%imports
import xml.etree.ElementTree as ET
import pandas as pd
#%%load
# parse an xml file by name
tree = ET.parse('Data/DFL-MAT-003CPF_full_video_labels_uniform_setpieces_2.5_0.4_1.0_214.5_4074.45.xml')

#%%#%%make dictionary to later transform into .csv
events_dict = {}

#%%
##find all events (instances) in the xml
lst = tree.findall('ALL_INSTANCES/instance')
print('Event Count:', len(lst))

#counter to keep track of event time in trimmed video
event_counter = 0
#match_id = 

#add info about event in dictionaries similar to the style of NetVlad ++ 
for event in lst:
    
    
    event_id = event.find('ID').text
    event_dict = {}

    event_dict["start_time_snippet"] = event.find('start').text
    event_dict["end_time_snippet"] = event.find('end').text
    event_dict["event_type"] = event.find("code").text
    
    event_dict['event_timestamp'] = 'None'

    
    #if an event exists in clip (and therefore also a timestamp), write in dict
    labels = event.findall("label")
    for label in labels:
        if label.find("group").text == 'Event_Timestamp':
            event_dict['event_timestamp'] = label.find('text').text
    
    event_counter += 1
    events_dict[event_id] = event_dict

#%%transform to df and save as .csv

df = pd.DataFrame.from_dict(data=events_dict, orient='index')
 
df.to_csv('Data/labels_kaggle.csv', header=True)

