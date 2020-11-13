import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sys

h_topics = t_topics = 8
directory = '/home/rajsuryan/Desktop/PopEvol_1960-2020/'
data_path = directory + 'Data/'
result_path = directory + 'Results/'
dtp = ""                            #String to add to filenames
#Parse command line arguments
if len(sys.argv) > 1:
    h_topics = int(sys.argv[1])
    t_topics = int(sys.argv[2])
    dtp = sys.argv[3]

#Load data
fname = data_path + "hot100_topics_" + str(h_topics) + "H" + str(t_topics) + "T" + dtp + ".csv"
topics = pd.read_csv(fname)


#Plot all the topics vs time (mean values with 95 percent CI)

#H-Topics

h_rows = (h_topics+1)//2          #number of rows in the plot

h_fig, ax = plt.subplots(2,h_rows, figsize = (20,7))
h_fig.suptitle = 'Evolution of H-Topics'
for i in range (1,h_topics+1):
    if i <(h_rows + 1):
        b = 0
        a = i
    else:
        b = 1
        a = i- h_rows
    y = "H-Topic:" + str(i)
    sns.lineplot(data = topics, x= "Year", y = y, ax = ax[b][a-1], palette=['red'])
    ax[b][a-1].set_ylim((0,0.5))

fname = result_path + "H-Topic_Evolution_" + str(h_topics) + "H" + str(t_topics) + "T" + dtp + ".png"
h_fig.savefig(fname)


#T-Topics
t_rows = (t_topics+1)//2          #number of rows in the plot
t_fig, ax = plt.subplots(2,t_rows, figsize = (20,7))
t_fig.suptitle = 'Evolution of T-Topics'
for i in range (1,t_topics+1):
    if i < (t_rows + 1):
        b = 0
        a = i
    else:
        b = 1
        a = i - t_rows
    y = "T-Topic:" + str(i)
    sns.lineplot(data = topics, x= "Year", y = y, ax = ax[b][a-1])
    ax[b][a-1].set_ylim((0,0.6))

fname = result_path + "T-Topic_Evolution_" + str(h_topics) + "H" + str(t_topics) + "T" + dtp + ".png"
t_fig.savefig(fname)

print("Plots saved to the Results Directory.")