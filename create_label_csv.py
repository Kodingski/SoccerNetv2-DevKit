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
lst = tree.findall('ALL_INSTANCES/instance')

#%%
#make dictionary to later transform into .csv
events_dict = {}

#counter to keep track of event time in trimmed video
event_counter = 0
#match_id = 


#add info about event in dictionaries similar to the style of NetVlad ++ 
for event in lst:
    
    starting_time = event_counter*2.5
    
    event_dict = {}

    event_dict["start_time_snippet"] = event.find('start').text
    event_dict["end_time_snippet"] = event.find('end').text
    
    event_id = event_counter+1
    #event_dict["start"] = starting_time
    #event_dict["end"] = starting_time + 2.5
    
    event_dict["event_attributes"] = []
    

    #initialiaze timesteps of actual events as none for the case none happened 
    #event_dict['timestep_raw'] = None
    event_dict['event_timestamp'] = None

    
    #if an event exists in clip (and therefore also a timestamp), write in dict
    labels = event.findall("label")
    for label in labels:
        if label.find("group").text == 'Event_Timestamp':
            #print(label.find('text').text)
            event_dict['event_timestamp'] = label.find('text').text
            #event_dict['timestep'] = starting_time + (float(event_dict['timestep_raw']) - float(event_dict["start_raw"])) 
        
        if label.find("group").text in ["tackle_WinnerResult", 
                                           "Event_SubType"] :
            event_dict["event_attributes"].append(label.find('text').text)
           
    event_type = event.find("code").text 
    
    
    if event_type == "Play":
        event_dict["event_attributes"].append("Open")
    
    if event_type in ['Freekick', 'Corner']:
        event_dict["event_attributes"].append(event_type)
        event_type = 'Play'        

    event_dict["event_type"] = event_type

    event_counter += 1
    events_dict[event_id] = event_dict

#%%transform to df and save as .csv
#reorder



df = pd.DataFrame.from_dict(data=events_dict, orient='index')
df = (df[['event_type', 'event_attributes', 
         'start_time_snippet', 'end_time_snippet',
         'event_timestamp']])

df.to_csv('Data/labels_kaggle.csv', header=True)

