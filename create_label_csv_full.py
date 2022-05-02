# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:44:54 2022

@author: timsc
"""
#%%imports
import xml.etree.ElementTree as ET
import pandas as pd
#%%create game list to create csv of
# parse an xml file by name, you just need to adjust the match_list 
match_list = ['DFL-MAT-003CPD_195.71_3944.95',
              'DFL-MAT-003CPE_193.75_3941',
              'DFL-MAT-003CPF_214.5_4074.45',
              'DFL-MAT-003CPG_188.95_4046.95',
              'DFL-MAT-003CPH_172.15_4103.5',
              'DFL-MAT-003CPI_196.4_3973.7',
              'DFL-MAT-003CPJ_237.3_4046.75',
              'DFL-MAT-003CPK_225.8_4256.5',
              'DFL-MAT-003CPL_197.65_3963.4']

#%%func
def create_label_csv(match_name):
    
    
    kickoff_ht1 = float(match_name.split('_')[1]) #188.95
    kickoff_ht2 = float(match_name.split('_')[2])
    #load data and define tree
    tree = ET.parse(rf'data/final-snippets/xml/{match_name}_final_snippets.xml')
    lst = tree.findall('ALL_INSTANCES/instance')
    
    #dictionary to remap challenge definitions to catalogue
    remap_challenges = {'dribbledAround' : 'opponent_rounded',
      'layoff': 'challenge_during_ball_transfer',
      'ballClaimed': 'opponent_dispossessed',
      'ballControlRetained':'possession_retained',
       'ballcontactSucceeded':'ball_action_forced',
       'fouled' : 'fouled' }
    
    #make dictionary to later transform into .csv
    events_dict = {}
    
    #counter to keep track of event time in trimmed video
    event_counter = 0
    
    #add info about event in dictionaries similar to the style of NetVlad ++ 
    for event in lst:
        
        starting_time = event_counter*2.5
        start_time_raw = float(event.find("start").text) 
        end_time_raw = float(event.find("end").text)
        event_dict = {}
    
        #event_dict["start_time_snippet"] = starting_time
        #event_dict["end_time_snippet"] = starting_time + 2.5
        
        
        event_id = event_counter+1
        event_dict["clip_start"] = start_time_raw
        event_dict["clip_end"] = end_time_raw
        
        event_dict["event_attributes"] = []
        
    
        #initialiaze timesteps of actual events as none for the case none happened 
        #event_dict['timestep_raw'] = None
        event_dict['event_timestamp'] = None
    
        
        #if an event exists in clip (and therefore also a timestamp), write in dict
        labels = event.findall("label")
        for label in labels:
            if label.find("group").text == 'Event_Timestamp':
                #print(label.find('text').text)
                event_dict['event_timestamp'] = float(label.find('text').text)
                #event_dict['timestep'] = starting_time + (float(event_dict['timestep_raw']) - float(event_dict["start_raw"])) 
            
            if label.find("group").text == "Event_SubType" :
                event_dict["event_attributes"].append(label.find('text').text)
                
            if label.find("group").text  == 'tackle_WinnerResult':
                event_dict["event_attributes"].append(remap_challenges[label.find('text').text])
               
        event_type = event.find("code").text 
        if event_type == 'TacklingGame':
            event_type = 'Challenge'
        #add gametime timestamp
        #if float(event_dict["start_time_snippet"]) < kickoff_ht2:
        #    event_dict['halftime'] = 1
        #    event_dict['start_time_snippet_halftime'] = float(event_dict['start_time_snippet']) - kickoff_ht1
        #    
        #    #event_dict['timestamp_halftime'] = float(event_dict['event_timestamp']) - kickoff_ht1
        #else:
        #    event_dict['halftime'] = 2
        #    #event_dict['timestamp_halftime'] = float(event_dict['event_timestamp']) - kickoff_ht2  
        #    event_dict['start_time_snippet_halftime'] = float(event_dict['start_time_snippet']) - kickoff_ht2
        
        if event_type == "Play":
            event_dict["event_attributes"].append("OpenPlay")
        
        if event_type in ['Freekick', 'Corner']:
            event_dict["event_attributes"].append(event_type)
            event_type = 'Play'        
    
        event_dict["event_type"] = event_type
    
        event_counter += 1
        events_dict[event_id] = event_dict
    
    #transform to df and save as .csv
    df = pd.DataFrame.from_dict(data=events_dict, orient='index')
    #reorder
    df = (df[['event_type', 'event_attributes', 'clip_start', 'clip_end',
             'event_timestamp']])
    
    df.to_csv(rf'data/final-snippets/full-video/csv/{match_name}_final_snippets.csv', header=True)

#%%
for match_name in match_list:
    create_label_csv(match_name)