import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

smoothening = 0
n_topics = 8
directory = '/home/rajsuryan/Desktop/PopEvol_1960-2020/'
data_path = directory + 'Data/'
result_path = directory + 'Results/'
if smoothening:
    topics = pd.read_csv(data_path + "hot100_topics.csv")
else:
    topics = pd.read_csv(data_path + "hot100_topics_without_smoothening.csv")

topics.drop(columns =  ["Unnamed: 0", "Song", "Performer", "Month"], inplace = True)


#Plot all the topics vs time (mean values with 95 percent CI)

#H-Topics

n_rows = (n_topics+1)//2          #number of rows in the plot

h_fig, ax = plt.subplots(2,4, figsize = (20,7))
h_fig.suptitle = 'Evolution of H-Topics'
for i in range (1,n_topics+1):
    if i <(n_rows + 1):
        b = 0
        a = i
    else:
        b = 1
        a = i- n_rows
    y = "H-Topic:" + str(i)
    sns.lineplot(data = topics, x= "Year", y = y, ax = ax[b][a-1], palette=['red'])
    ax[b][a-1].set_ylim((0,0.5))

if smoothening:
    h_fig.savefig(result_path + "H-Topic_Evolution")
else:
    h_fig.savefig(result_path + "H-Topic_Evolution_without_smoothening")


#T-Topics
t_fig, ax = plt.subplots(2,n_rows, figsize = (20,7))
t_fig.suptitle = 'Evolution of T-Topics'
for i in range (1,n_topics+1):
    if i < (n_rows + 1):
        b = 0
        a = i
    else:
        b = 1
        a = i - n_rows
    y = "T-Topic:" + str(i)
    sns.lineplot(data = topics, x= "Year", y = y, ax = ax[b][a-1])
    ax[b][a-1].set_ylim((0,0.6))
if smoothening:
    t_fig.savefig(result_path + "T-Topic_Evolution")
else:
    t_fig.savefig(result_path + "T-Topic_Evolution_without_smoothening")

print("Plots saved to the Results Directory.")