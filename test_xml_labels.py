# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:29:07 2022

@author: timsc
"""

#%% imports
from xml.dom import minidom

#%%load
# parse an xml file by name
file = minidom.parse('Data/DFL-MAT-003CPF_full_video_labels_uniform_setpieces_2.5_0.4_1.0_214.5_4074.45.xml')



print('\nAll model data:')
for elem in models:
  print(elem.firstChild.data)