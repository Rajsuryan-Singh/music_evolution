import pandas as pd
import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from joblib import Parallel, delayed
import copy
import sys
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

data_path = '/home/rajsuryan/Desktop/PopEvol_1960-2020/Data/'
preview_dir = data_path + 'Song Previews/'

#Function to load songs given spotify ID
def get_song(song_id, preview_dir):
    file = preview_dir + str(song_id) + ".mp3"
    y, sr = librosa.load(file)
    return y, sr

# Band Pass Filter (67Hz, 6000Hz)
def butter_bandpass(lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass(data, sr, order=5):
    lowcut = 67.0
    highcut = 6000.0
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    y = lfilter(b, a, data)
    return y

def get_timbre_features(id):
    y, sr = get_song(id)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, n_fft = 1024, hop_length = 512)
    zcc = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=512)
    mfcc0 = mfcc[0,:]
    del_mfcc = librosa.feature.delta(mfcc0).reshape(zcc.shape)
    coeff = np.vstack([mfcc, del_mfcc, zcc])
    return coeff

timbre_features = dict()
test = songs.iloc[:20]
start_time = time.time()
timbre_features_arr = Parallel(n_jobs=16, prefer="threads", 
                                verbose = 5)(delayed(get_timbre_features)(i) for i in songs["id"])

