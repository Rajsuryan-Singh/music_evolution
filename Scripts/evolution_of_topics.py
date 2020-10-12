import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

directory = '/home/rajsuryan/Desktop/PopEvol_1960-2020/'
data_path = directory + 'Data/'
result_path = directory + 'Results/'
topics = pd.read_csv(data_path + "hot100_topics.csv")
topics.drop(columns =  ["Unnamed: 0", "Song", "Performer", "Month"], inplace = True)


#Plot all the topics vs time (mean values with 95 percent CI)

#H-Topics

h_fig, ax = plt.subplots(2,4, figsize = (20,7))
h_fig.suptitle = 'Evolution of H-Topics'
for i in range (1,9):
    if i <5:
        b = 0
        a = i
    else:
        b = 1
        a = i-4
    y = "H-Topic:" + str(i-1)
    sns.lineplot(data = topics, x= "Year", y = y, ax = ax[b][a-1], palette=['red'])
    ax[b][a-1].set_ylim((0,0.5))

h_fig.savefig(result_path + "H_Topics_Evolution")

#T-Topics
t_fig, ax = plt.subplots(2,4, figsize = (20,7))
t_fig.suptitle = 'Evolution of T-Topics'
for i in range (1,9):
    if i <5:
        b = 0
        a = i
    else:
        b = 1
        a = i-4
    y = "T-Topic:" + str(i-1)
    sns.lineplot(data = topics, x= "Year", y = y, ax = ax[b][a-1])
    ax[b][a-1].set_ylim((0,0.6))

t_fig.savefig(result_path + "T-Topic_Evolution")

print("Plots saved to the Results Directory.")