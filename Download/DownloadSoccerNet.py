import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

#%%
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="")


mySoccerNetDownloader.downloadGames(files=["Labels-v2.json", "1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=[
                                    "train", "valid", "test", "challenge"])  # download Features
