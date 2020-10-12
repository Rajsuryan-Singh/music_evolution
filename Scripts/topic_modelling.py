from sklearn.decomposition import LatentDirichletAllocation as LDA
import pandas as pd
import matplotlib as plt
import re

data_path = '/home/rajsuryan/Desktop/PopEvol_1960-2020/Data/'
data = pd.read_csv(data_path + 'hot100_all_features.csv')

#Get column names for timbre and harmony features
columns = data.columns
timbre_col = [col for col in columns if re.search("Timbre", col) ]
harm_col = [col for col in columns if re.search("Harm", col)]

#Create matrices of timbre and harmony features
timbre_features = data[timbre_col]
timbre_features = timbre_features.to_numpy()

harm_features = data[harm_col]
harm_features = harm_features.to_numpy()

#Apply LDA on both separately
lda_timbre = LDA(n_components=8, random_state=0)
lda_timbre.fit(timbre_features)
topics_timbre = lda_timbre.transform(timbre_features)

lda_timbre = LDA(n_components=8, random_state=0)
lda_timbre.fit(harm_features)
topics_harm = lda_timbre.transform(harm_features)

#Assemble topics in a dataframe
df_harm = pd.DataFrame(topics_harm, columns = ["H-Topic:" + str(i) for i in range(1,9)])
df_timbre = pd.DataFrame(topics_timbre, columns = ["T-Topic:" + str(i) for i in range(1,9)])
df_topics = pd.concat([df_timbre, df_harm], axis=1)

#Add song title, artist and date
col_to_keep = ["Song", "Performer", "Year", "Month"] 
temp_df = data[col_to_keep]
df_topics = pd.concat([temp_df, df_topics], axis = 1)

#Save the topics dataframe
df_topics.to_csv(data_path + "hot100_topics.csv")

print("Topic Modelling Done!")