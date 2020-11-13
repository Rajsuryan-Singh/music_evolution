import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from wordcloud import WordCloud
import sys
directory = '/home/rajsuryan/Desktop/PopEvol_1960-2020/'
data_path = directory + 'Data/'
results_path = directory + 'Results/'

"""
The command line arguments for this script (starting with index 1) are as follows:
1. Number of h_topics
2. Number of t_topics
3. doc_topic_prior used in topic modelling
"""

h_topics = t_topics = 8
dtp = ""                            #String to add to filenames
#Parse command line arguments
if len(sys.argv) > 1:
    h_topics = int(sys.argv[1])
    t_topics = int(sys.argv[2])
    dtp = sys.argv[3]

#Load data
fname = data_path + "hot100_topics_" + str(h_topics) + "H" + str(t_topics) + "T" + dtp + ".csv"
songs = pd.read_csv(fname)
songs = songs[~songs["Tags"].isna()]

#Separate T-Topics
columns = songs.columns
timbre_topics = [i for i in columns if i.startswith("T-Topic")]
timbre = songs[timbre_topics]
timbre = timbre.to_numpy()

#Get most highly represented topics
max_topic = np.argmax(timbre, axis = 1)
max_topic= [i + 1 for i in max_topic]
songs["Top T-Topic"] = max_topic

#Clean up tags and prepare for wordcloud (remove spaces and generate counts)
corrected = []
for tags in songs["Tags"]:
    tags = tags.split(",")
    tags = " ".join(["-".join(i.split(" ")) for i in tags])
    corrected.append(tags)

songs["Tags"] = corrected

#Generate tag counts per topic
tags_per_topic = dict()
for i in range(1,t_topics+1):
    df = songs[songs["Top T-Topic"] == i]
    counts = dict()
    for tag in df["Tags"]:
        counts[tag] = counts.get(tag,0) +1
    tags_per_topic[i] = counts
tags_per_topic[1]

# Remove noise (subtract the min count for the tags found in all topics)
common = set(tags_per_topic[1].keys())
sets = [set(tags_per_topic[i].keys()) for i in range(2,t_topics+1)]
for i in sets:
    common = common.intersection(i)
common = {i:float('Inf') for i in common}
for i in tags_per_topic:
    for tag in common:
        common[tag] = min(common[tag], tags_per_topic[i][tag])
for i in tags_per_topic:
    for tag in common:
        tags_per_topic[i][tag] -= common[tag]


#Generate Wordcloud
n_rows = (t_topics+1)//2
fig, ax = plt.subplots(2,n_rows, figsize = (20,7))
fig.suptitle = 'Distribution of tags over T-Topics'


for i in range (1,t_topics+1):
    wordcloud = WordCloud(background_color='black',
                          colormap = "Pastel2",
                          collocations=False).generate_from_frequencies(tags_per_topic[i])
    if i <(n_rows + 1):
        b = 0
        a = i
    else:
        b = 1
        a = i- n_rows
    ax[b][a-1].imshow(wordcloud)
    ax[b][a-1].set_title("T-Topic" + str(i))
    ax[b][a-1].axis("off")
fname = results_path + 'tags_wordcloud_' + str(h_topics) + "H" + str(t_topics) + "T" + dtp + ".png"
fig.savefig(fname)

print("The tag-wordclouds have been saved in the results directory.")