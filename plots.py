# -*- coding: utf-8 -*-
"""
last update 9/4/2021
@author: Tonatiuh Hernández-Del-Toro (tonahdztoro@gmail.com)

Code for the paper 
"Toward asynchronous EEG-based BCI: Detecting imagined words segments in continuous EEG signals"
published on the journal "Biomedical Signal Processing and Control"

arXiv: 
doi: 10.1016/j.bspc.2020.102351


This file collects the function used to plot the figures
"""


import os
import pandas as pd
import numpy as np
import seaborn as sns


# Dataset results to be plotted
ds = 1


# Set the global directory
script_dir = os.path.dirname(__file__)
results_path = 'Results/DS' + str(ds) + '/'
file_path = os.path.join(script_dir, results_path)

# Set the feature sets to plot:
# For the first three feature sets (IWE, EMD, GHE) set FeatureSets = 3
# For the last two feature sets (All, PCA) set FeatureSets = 2
FeatureSets = 3

if FeatureSets == 3:
    FS = ['IWE', 'EMD', 'GHE']
else:
    FS = ['All', 'PCA']

CLF = ['RF', 'kNN', 'SVM', 'LogReg']


F1scores = {} #Initializes the F1scores struct

for clf in CLF:

    for fs in FS:
        
        container = np.load(file_path + fs + '_' + clf  + '_DS' + str(ds) + '.npz', mmap_mode='r');
        data = [container[key] for key in container]
        F1score = data[0]
        F1scoreMean = np.mean(F1score, axis=(1,2))
        F1scores[fs] = F1scoreMean

    if clf == 'RF':
        df = pd.DataFrame.from_dict(F1scores)
        df['clf'] = clf
    else:
        df1 = pd.DataFrame.from_dict(F1scores)
        df1['clf'] = clf
        df = pd.concat([df,df1])   
        

# Create the dataframe
df_new = df.reset_index()
df_new = pd.melt(df,id_vars=['clf'],var_name='FS', value_name='Precision')


# Do the box plot

plt = sns.boxplot(x="clf", y='Precision', hue="FS", data=df_new, showmeans=True, palette="Set1", \
                  meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
plt.set(ylim=(0.4, 1.01))
plt.set_ylabel('F1 score')
plt.set_xlabel('Classifiers')
plt.set_title('F1 score for all feature sets with all classifiers')
plt.yaxis.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), fancybox=True, shadow=True, ncol=3)

fig = plt.get_figure()
del plt

# Save the figure
figures_path = 'Figures/'
complete_path = os.path.join(script_dir, figures_path)
fig.savefig(complete_path + 'DS' + str(ds) + '_' + str(FeatureSets) + '.pdf', dpi=400, bbox_inches="tight")





