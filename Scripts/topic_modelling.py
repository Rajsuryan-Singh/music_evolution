from sklearn.decomposition import LatentDirichletAllocation as LDA
import pandas as pd
import matplotlib as plt
import re
import sys

#Setting up modelling parameters based on command line arguments

h_topics = t_topics =  8
doc_topic_prior = None
dtp = ""               #String to add to filenames
if len(sys.argv) > 1:
    h_topics = int(sys.argv[1])
    t_topics = int(sys.argv[2])
    doc_topic_prior = float(sys.argv[3])
    dtp = sys.argv[3]

data_path = '/home/rajsuryan/Desktop/PopEvol_1960-2020/Data/'

data = pd.read_csv(data_path + 'hot100_features_with_tags.csv')


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
lda_timbre = LDA(n_components=t_topics, doc_topic_prior=doc_topic_prior, random_state=0)
lda_timbre.fit(timbre_features)
topics_timbre = lda_timbre.transform(timbre_features)

lda_harm = LDA(n_components=h_topics, doc_topic_prior=doc_topic_prior, random_state=0)
lda_harm.fit(harm_features)
topics_harm = lda_harm.transform(harm_features)

#Summary of the top components of topics 

def get_top_components(model, feature_names, n):
    #n is the number of top components you want to output
    n_topics = model.n_components
    d = {"Topic": ["Topic:" + str(i) for i in range(1,n_topics+1)],
                                     "Top Components":["" for i in range(1,n_topics+1)] }
    results = pd.DataFrame(d)

    for topic_idx, topic in enumerate(model.components_):
        results["Top Components"][topic_idx] = ", ".join([feature_names[i] for i in topic.argsort()[:-n - 1:-1]])
    return results

top_components_timbre = get_top_components(lda_timbre, timbre_col, 4)




fname = data_path + "Top Components LDA Timbre_" + str(h_topics) + "H" + str(t_topics) + "T" + dtp + ".csv"
top_components_timbre.to_csv(fname)

top_components_harm = get_top_components(lda_harm, harm_col, 4)

fname = data_path + "Top Components LDA Harmony_" + str(h_topics) + "H" + str(t_topics) + "T" + dtp + ".csv"
top_components_harm.to_csv(fname)

#Assemble topics in a dataframe
df_harm = pd.DataFrame(topics_harm, columns = ["H-Topic:" + str(i) for i in range(1,h_topics + 1)])
df_timbre = pd.DataFrame(topics_timbre, columns = ["T-Topic:" + str(i) for i in range(1,t_topics+1)])
df_topics = pd.concat([df_timbre, df_harm], axis=1)

#Add song title, artist and date
col_to_keep = ["Song", "Performer", "Year", "Month", "Tags"] 
temp_df = data[col_to_keep]
df_topics = pd.concat([temp_df, df_topics], axis = 1)

#Save the topics dataframe
fname = data_path + "hot100_topics_" + str(h_topics) + "H" + str(t_topics) + "T" + dtp + ".csv"
df_topics.to_csv(fname)


print("Topic Modelling Done!")