import pandas as pd
import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
from collections import deque
from sklearn import preprocessing
from joblib import Parallel, delayed
import copy
from scipy.signal import butter, lfilter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


data_path = '/home/rajsuryan/Desktop/PopEvol_1960-2020/Data/'
preview_dir = data_path + 'Song Previews/'
#Load Song
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

#Get and Plot Chroma  
def get_chroma_cqt(y, sr):
    y_harm = librosa.effects.harmonic(y=y, margin=8)
    chroma_os_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=12*3, hop_length = 1024)
    # 3 bins per octave also used in the nnls chroma
    # hop length of 1024 used in this paper https://royalsocietypublishing.org/doi/10.1098/rsos.150081#d3e991
    
    chroma_filter = np.minimum(chroma_os_harm,
                               librosa.decompose.nn_filter(chroma_os_harm,
                                                           aggregate=np.median,
                                                           metric='cosine'))
    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 20))
    # The above two operations are taken from librosa documentation of enhanced chroma
    return chroma_smooth


def plot_chroma(chroma):
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis = 'time')
    plt.colorbar()
    plt.ylabel('Notes')


# H-Lexicon

# Build chord templates
types = ["M", "m", "7", "m7"]
notes = [ "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A","Bb", "B"]
chords = [a + "." + b for a in notes for b in types]

templates = {
    "M" : [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "m" : [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "7" : [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "m7" : [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
}

# Initialise templates
CT = np.zeros((len(chords), len(notes)), dtype = float)

# Fill up the templates

for idx, chord in enumerate(chords):
    [root, chord_type] = chord.split(".")
    dist_from_A = notes.index(root)

    """
    There are two parameters in a chord - root and chord type. 
    Any template can be derived from the template for chords based on A by transposing them.
    The transposition can be acheived by rotating the template for A. 
    Transposing up x semitones = rotating the array clockwise by x units (hence np.roll(template, x))
    """
    
    CT[idx,:] = np.roll(templates[chord_type], dist_from_A )
    
# Vectorised Chord Transcription

def get_likely_chords(chroma, CT, chords):

    # Step 1
    CT_std = preprocessing.scale(CT, axis = 1)
    chroma_std = preprocessing.scale(chroma, axis = 0)

    # Step 2
    M = np.dot(CT_std, chroma_std)/11

    # Step 3
    M_smooth = scipy.ndimage.median_filter(M, size = (1, 43))

    # Step 4 
    chord_id = np.argmax(M_smooth, axis = 0)
    likely_chords = [chords[i] for i in chord_id]
    corr = np.max(M_smooth, axis = 0)
    
    return likely_chords, corr

# Labelling the chord changes
def label_change(C1, C2, notes):
    C1 = C1.split(".")
    C1_root = C1[0]
    C1_type = C1[1]
    
    C2 = C2.split(".")
    C2_root = C2[0]
    C2_type = C2[1]
    
    dist = notes.index(C2_root) - notes.index(C1_root)
    if dist<0: dist+=12
        
    label = C1_type + "." + str(dist) + "." + C2_type
    return label
 
def get_h_lex(likely_chords, corr, num_frames, notes):
    h_lex = {"NA":0}
    types = ["M", "m", "7", "m7"]

    for i in types:
        for j in types:
            for k in range(12):
                change = i + "." + str(k) + "." + j
                h_lex[change] = 0

    for i in range(len(likely_chords) - num_frames):
        C1 = likely_chords[i]
        C2 = likely_chords[i + num_frames]
        change = label_change(C1, C2, notes)
        if corr[i] + corr[i+num_frames] > 0.8:  # average correlation greater than 0.4
            h_lex[change] +=1
        else:
            h_lex["NA"] +=1
    return h_lex

#Function to filter spurious chord detections
def mode(arr):
    return (scipy.stats.mode(arr)[0][0])

def remove_spurious_chords(likely_chords, nf):
    n = len(likely_chords)    
    fixed = [mode(likely_chords[i-nf//2: i+nf//2]) for i in range(nf//2+1, n- nf//2)]
    return fixed

# The main function to populate the H-Lexicon in the dataframe using the auxillary functions

def main_hlex(songs, CT, chords, preview_dir):
    #Initialising new columns in the dataframe
    new_columns = {"NA":[]}
    types = ["M", "m", "7", "m7"]

    for i in types:
        for j in types:
            for k in range(12):
                change = "Harm: " + i + "." + str(k) + "." + j
                new_columns[change] = []
                
    for i in tqdm(range(songs.shape[0])):
        song_id = songs.iloc[i]["id"]
        y, sr = get_song(song_id, preview_dir)                
        y = bandpass(y, sr, order = 6)
        chroma = get_chroma_cqt(y, sr) 
        chroma_sr = chroma.shape[1]//30        
        likely_chords, corr = get_likely_chords(chroma, CT, chords)  

        #Tempo detection
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        
        num_frames = int(chroma_sr*60/tempo)         # Quarter note
        # num_frames = int(chroma_sr * 30/tempo)           # Eigth note
        #Remove spurious chords with window size of two quarter notes
        likely_chords = remove_spurious_chords(likely_chords, 2*num_frames)
        h_lex = get_h_lex(likely_chords, corr, num_frames, notes)

        for j in h_lex.keys():                                # Questionable Scalability! 
            new_columns[j].append(h_lex[j])

    songs= songs.assign(**new_columns)
    return songs

songs = pd.read_csv(data_path + 'hot100_with_tlex.csv')
songs = songs[songs["availability"] == 1]
songs_final = main_hlex(songs, CT, chords, preview_dir)
songs_final.to_csv(data_path + "hot100_all_features.csv")
print("Calculation of H-Lexicon done!")