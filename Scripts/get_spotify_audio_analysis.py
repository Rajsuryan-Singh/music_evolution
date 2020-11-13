import pandas as pd 
import numpy as np 
import spotipy
from spotipy import SpotifyClientCredentials
import os
import pickle
import swifter
from tqdm import tqdm

# Change this to the location of your project directory
data_path = '/home/rajsuryan/Desktop/PopEvol_1960-2020/Data/'

#Setup API credentials
client_credentials_manager = SpotifyClientCredentials(client_id="8b6b3fab9bb04c17ab4aa187c5dd826b",client_secret="833badffedd8400293eb09d35c096455")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=20, retries=3 )

def get_ID(track, artist):
    #Search for Spofity song ID 
    songs=sp.search(q='track:'+track+ ' ' + 'artist:' + artist+ '*' , type='track')
    results = songs['tracks']['items']
    if len(results) ==0:
        return(0)
    else:
        track = results[0]
        if results[0]["artists"][0]['name'].lower() == artist.lower():
            song_id = str(track["id"])
            return song_id
        else:
            return(0)


#Load the file with all spotify IDs (create it if it doesn't exist)
id_fname = data_path + 'hot100_IDs.csv'

if not os.path.exists(id_fname):
    hot100 = pd.read_csv(data_path + 'hot100.csv', encoding='latin1')
    hot100["id"] = hot100.swifter.apply(lambda row: get_ID(row["Song"],row["Performer"]) ,axis=1)
    hot100 = hot100[hot100["id"] != 0]
    hot100.to_csv(id_fname)
else:
    hot100 = pd.read_csv(id_fname)

#Setup directory and intialise
analysis_dir = data_path + "audio_analysis/"
if not os.path.exists(analysis_dir):
    os.mkdir(analysis_dir)

#Check if the script has been run before
if "Analysis Availability" in hot100.columns:
    analysis_avail = np.array(hot100["Analysis Availability"])
else:
    analysis_avail = np.zeros(hot100.shape[0], dtype = int)

#Retrieve data from the API
for i in tqdm(range(hot100.shape[0])):
    ID = hot100.iloc[i]["id"]
    if analysis_avail[i] == 0:
        try:
            analysis = sp.audio_analysis(ID)
            fname = analysis_dir + ID + ".pkl"
            with open(fname, "wb") as filehandler:
                pickle.dump(analysis, filehandler)
            analysis_avail[i] = 1
        except:
            analysis_avail[i] = 0

#Add analysis availability to dataframe
hot100["Analysis Availability"] = analysis_avail

#Save 
hot100.to_csv(data_path + "hot100_IDs.csv")

print("\nAudio Analysis info retrieved from Spotify!")
print(sum(analysis_avail),  " out of ", len(analysis_avail), " have been found.")