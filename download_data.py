# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 10:26:59 2022

@author: timsc
"""

#%%imports
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader


#%%initialize downloader and get features
path = 'data'
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=path)

mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])