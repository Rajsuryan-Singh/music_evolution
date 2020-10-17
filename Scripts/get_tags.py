import pandas as pd 
from tqdm import tqdm
import requests
from requests.adapters import TimeoutSauce
import os
import backoff




import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session



directory = '/home/rajsuryan/Desktop/PopEvol_1960-2020/'
data_path = directory + "Data/"
api_file = directory + "last.fm_api_info.txt"           #replace with the file having your api key
api_key = open(api_file).read().strip()

#Load the dataframe with all the songs
songs_df = pd.read_csv(data_path + 'hot100_topics.csv')
songs = songs_df["Song"]
artists = songs_df["Performer"]

#Initiate search payload
payload = {
        'api_key': api_key,
        'method': 'track.getTopTags',
        'format': 'json',
        'track': '',
        'artist': ''
    }

#Search for songs and retrieve 
all_tags = []
for i in tqdm(range(len(songs))):

    payload['track'] = songs[i]
    payload['artist'] = artists[i]
    try:
        r = requests_retry_session().get('http://ws.audioscrobbler.com/2.0/',  params=payload)
    except Exception:
        all_tags.append("")
    else:
        if 'toptags' in r.json().keys():
            tag_json = r.json()['toptags']['tag']
            tags = [tag['name'] for tag in tag_json[:5] if tag['count']>20]
            
            all_tags.append(", ".join(tags))
        else:
            all_tags.append("")

songs_df["Tags"] = all_tags

songs_df.to_csv(data_path + 'hot100_topics_with_tags.csv')

print("Tags have been retrieved from last.fm!")
