import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import sys
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle


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

#Setting up parameters
topics = [ 4, 6, 8, 10, 12, 14, 16]
doc_topic_prior = [0.001, 0.1, 1, 10]
topic_word_prior = [0.001, 0.1, 1, 10]

cv_results_h = np.zeros([len(topics), len(doc_topic_prior)])
cv_results_t = np.zeros([len(topics), len(doc_topic_prior)])


for i in tqdm(range(3)):
    train_h, test_h = train_test_split(harm_features, test_size = 0.3, random_state = i, shuffle = True)
    train_t, test_t = train_test_split(timbre_features, test_size = 0.3, random_state = i, shuffle = True)

    #Check CV score for both with varying parameters
    for i, n_topics in enumerate(topics): 
        for j, alpha in enumerate(doc_topic_prior):
            lda_h = LDA(n_components = n_topics, 
                            doc_topic_prior = alpha,
                            random_state = 0, 
                            n_jobs = -1)
            lda_t = LDA(n_components = n_topics, 
                            doc_topic_prior = alpha,
                            random_state = 0, 
                            n_jobs = -1)                                
            lda_h.fit(train_h)
            lda_t.fit(train_t)

            cv_results_h[i,j] = lda_h.score(test_h)
            cv_results_t[i,j] = lda_t.score(test_t)


file = open(data_path + 'lda_params_harmony.pickle', 'wb')
pickle.dump(cv_results_h, file)
file.close()

file = open(data_path + 'lda_params_timbre.pickle', 'wb')
pickle.dump(cv_results_t, file)
file.close()

print("Log likelihood scores over a range of parameters have been saved in the results directory!")
