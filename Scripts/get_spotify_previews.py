import pandas as pd
import numpy as np 
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import swifter
from tqdm import tqdm
import requests

quick_test = 0                     # Set to 1 if you quickly want to test if everything is working


# Change this to the location of your project directory
dir = '/home/rajsuryan/Desktop/PopEvol_1960-2020/'

#Load the billboard dataset
hot100 = pd.read_csv(dir + 'Data/hot100.csv',encoding='latin1')
if quick_test:
    hot100 = hot100.iloc[:20]                                      

#Setup API credentials
client_credentials_manager = SpotifyClientCredentials(client_id="8b6b3fab9bb04c17ab4aa187c5dd826b",client_secret="833badffedd8400293eb09d35c096455")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



#Function: searches for a song and returns Spotify ID if found
def get_ID(track, artist):
    #Search for Spofity song ID 
    songs=sp.search(q='track:'+track+' '+'artist:'+artist+'*' , type='track')
    results = songs['tracks']['items']
    if len(results) ==0:
        return(0)
    else:
        track = results[0]
        song_id = str(track["id"])
        return song_id
        
        

id_fname = dir + 'Data/hot100_IDs.csv'


if not os.path.exists(id_fname):
    hot100["id"] = hot100.swifter.apply(lambda row: get_ID(row["Song"],row["Performer"]) ,axis=1)
    hot100 = hot100[hot100["id"] != 0]
    hot100.to_csv(id_fname)


preview_fname = dir + 'Data/hot100_previews.csv'


if not os.path.exists(preview_fname):
    #Load the dataframe with spotify IDs
    hot100_ID = pd.read_csv(id_fname)

    # Get preview url and update availability (1 if preview url found, 0 otherwise)
    tracks = hot100_ID["id"]
    results = [sp.track(i) for i in tracks]

    preview_url = [i['preview_url'] for i in results]
    availability = [1 if i else 0 for i in preview_url]

    hot100_ID["preview_url"] = preview_url
    hot100_ID["availability"] = availability

    #Save
    hot100_ID.to_csv(preview_fname)


hot100_previews = pd.read_csv(preview_fname)
hot100_previews = hot100_previews[["id", "preview_url", "availability"]]
hot100_previews = hot100_previews[hot100_previews['availability'] == 1]



#Download mp3 files from preview urls
n = hot100_previews.shape[0]
for i in tqdm(range(n)): 
    url = hot100_previews.iloc[i]["preview_url"]
    r = requests.get(url, allow_redirects=True)
    fname = dir + 'Data/Song Previews/' + hot100_previews.iloc[i]["id"] + '.mp3'
    with open(fname, 'wb') as f:
        f.write(r.content)

print("All done!")



