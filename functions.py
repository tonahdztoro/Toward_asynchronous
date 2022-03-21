# -*- coding: utf-8 -*-
"""
last update 9/4/2021
@author: Tonatiuh Hernández-Del-Toro (tonahdztoro@gmail.com)

Code for the paper 
"Toward asynchronous EEG-based BCI: Detecting imagined words segments in continuous EEG signals"
published on the journal "Biomedical Signal Processing and Control"

arXiv:2105.04294
doi: 10.1016/j.bspc.2020.102351


This file collects the functions used in the script main
"""

import numpy as np
import os
import json
import math
import random

from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier





def Multiple(n, m , start): 
    """Creates an array of m multiples of n starting from start"""
    a = np.arange(start, (m * n), n)
    return a


def Get_indices(FeatureSet):
    """ Creates the indices of one of the 5 feature sets"""
    Idx = []
    if FeatureSet == 'IWE':
        for i in range(5):
            Idx = np.append( Idx, Multiple(20, 14, i) )
    elif FeatureSet == 'EMD':
        for i in range(12):
            Idx = np.append( Idx, Multiple(20, 14, i + 5) )  
    elif FeatureSet == 'GHE':
        for i in range(3):
            Idx = np.append( Idx, Multiple(20, 14, i + 17) ) 
    elif FeatureSet == 'All':
        Idx = np.arange(280)
    elif FeatureSet == 'PCA':
        Idx = np.arange(280)
    else:
        print('The selected feature set is not valid!')
    Idx = np.sort(Idx)
    Idx = Idx.astype(int)
    return Idx

def Extract_subject_folds(subject, script_dir, DS):
    """Extract the 4 folds of the selected subject and dataset
    In each fold is included the train and test sets"""
    rel_path = 'DataSets/DS' + str(DS) + '/S' + str(subject + 1) + '.txt'
    file_path = os.path.join(script_dir, rel_path)
    with open(file_path) as f:
        SubjectFolds = json.load(f)
    return SubjectFolds

def Select_classifier(clf):
    """Sets the selected classifier """
    if clf == 'RF':
        selected_clf = RandomForestClassifier(n_estimators=100)
    elif clf == 'kNN':
        selected_clf = KNeighborsClassifier(n_neighbors=100)
    elif clf == 'SVM':
        selected_clf = svm.SVC()
    elif clf == 'LogReg':
        selected_clf = LogisticRegression(max_iter = 10000)
    else:
        print('The selected classifier is not valid!')
    return selected_clf


def BalanceCorpus(Corpus):
    """Takes a corpus and outputs a corpus with balanced classes of 0 and 1"""
    
    # Creates an empy array of same lenght as features
    Instances_0 = np.zeros([Corpus.shape[1]])
    Instances_1 = np.zeros([Corpus.shape[1]])
    
    # Separates the 0 isntances from the 1 instances
    for i in range(Corpus.shape[0]):
        if Corpus[i,-1] == 0:
            Instances_0 = np.vstack((Instances_0, Corpus[i,:]))
        else:
            Instances_1 = np.vstack((Instances_1, Corpus[i,:]))
    
    # Removes first row because is zero
    Instances_0 = np.delete(Instances_0, (0), axis=0)
    Instances_1 = np.delete(Instances_1, (0), axis=0)
    
    # Finds which instances set is bigger, if 0 or 1.
    # Then takes the smaller set and fills it with random repetitions from the same set
    # to match the same number as the bigger set
    if Instances_1.shape[0] < Instances_0.shape[0]:
        dif = Instances_0.shape[0] - Instances_1.shape[0]
        Instances_1 = np.vstack((Instances_1, Instances_1[:dif,:]))
    elif Instances_1.shape[0] > Instances_0.shape[0]:
        dif = Instances_1.shape[0] - Instances_0.shape[0]
        Instances_0 = np.vstack((Instances_0, Instances_0[0:dif,:]))
    BalancedCorpus = np.vstack((Instances_0, Instances_1))
    
    return BalancedCorpus


def BalanceMultiTrials(Trials):
    """Takes as input a list of trials and outputs a balanced corpus X with all the 
    instances of each trial and its classes"""
    
    # Stacks all the intances from the set of trials
    for trial in range(len(Trials)):
        if trial == 0:
            Corpus = np.array(Trials[trial])
        else:
            Corpus = np.vstack((Corpus, np.array(Trials[trial])))
            
    # Balance the corpus to have the same number of 0 and 1 instances
    BalancedCorpus = BalanceCorpus(Corpus)
    
    return BalancedCorpus


def Remove_NaN(Train):
    """Converts all NaN values to 0"""
    for i in range(Train.shape[0]): 
        for j in range(Train.shape[1]): 
            if np.isnan(Train[i,j]) or  np.isinf(Train[i,j]):
                Train[i,j] = 0
    return Train


def round_half_up(n, decimals=0):
    """Rounds to 1 when 0.5"""
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def CalculateActualTrial(Epochs, NoSamples):
    """Creates a vector from the epochs, which represents the real trial segments' classes"""
    onset = int( round_half_up(Epochs[0]/128*10) - 1)
    ending = int( round_half_up(Epochs[1]/128*10) - 1)
    ActualTrial = np.zeros(NoSamples)
    ActualTrial[onset:ending] = 1
    return ActualTrial

def find_majority(votes):
    """Find majority vote on a vector of zeros and ones"""
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        return random.randint(0,1) # If it is a tie return a random 0 or 1
    return top_two[0][0]

def SmallerWindow(Predictions):
    """Function that takes the predictions of the classifier and returns a prediction vector according
    to the adjacent predictions. This means that exploits the information of the
    overlap of signals to create a virtual vector as if the windosize was equal
    the overlap"""
    Overlap = 13
    NoWindows = int(round_half_up(64/Overlap)) 
    NewPrediction = []
    for i in range(len(Predictions)):
        if i < NoWindows - 1:
            if i == 0:
                NewPrediction = np.append(NewPrediction,Predictions[i]);
            elif i == 1:
                NewPrediction = np.append(NewPrediction,Predictions[i]);
            else:
                NewPrediction = np.append(NewPrediction,find_majority(Predictions[:i]));
        else:
            NewPrediction = np.append(NewPrediction,find_majority(Predictions[i-NoWindows+1:i+1]));
    return NewPrediction


def ErrorCorrection(PredictionsSmallerWindow):
    """Makes a first neighbor correction, comapres each value with the neighbors and takes the majority"""
    CorrectedPredictions = [];
    for i in range(len(PredictionsSmallerWindow)):
        if i  == 0:
            CorrectedPredictions = np.append(CorrectedPredictions,PredictionsSmallerWindow[i]);
        elif i == len(PredictionsSmallerWindow) - 1:
            CorrectedPredictions = np.append(CorrectedPredictions,PredictionsSmallerWindow[i]);    
        else:
            CorrectedPredictions = np.append(CorrectedPredictions,find_majority(PredictionsSmallerWindow[i-1:i+2]));
    return CorrectedPredictions


def Evaluate_trial(Trial, Epochs, trial, scaler, Idx, FeatureSet, Clf, pca):
    """Takes a trial, attempts to predict it, applied the window size reduction and the error correction.
    Then compares it with the actual trial and returns the f1score, precisiona dn recall obtained on that trial"""
        
    # Scales the features to normal values
    Xtest = Trial[:,Idx] # Select only the features from the selected feature set
    Xtest = Remove_NaN(Xtest) # To correct NaN and inf values
    Xtest = scaler.transform(Xtest)
    
    
    # Applies PCA qhen selected
    if FeatureSet == 'PCA':
        Xtest = pca.transform(Xtest)
    
    Trial_pred = Clf.predict(Xtest) # Predicts the trial
    Trial_smaller = SmallerWindow(Trial_pred) # Window reduction
    Trial_corrected = ErrorCorrection(Trial_smaller) # Corrects isolated false positives and false negatives
    
    ActualTrial = CalculateActualTrial(Epochs, Xtest.shape[0]) # Get the correct predictions to compare
    
    # Get metrics
    TrialF1score = metrics.f1_score(Trial_corrected, ActualTrial, average='weighted')
    TrialPrecision = metrics.recall_score(Trial_corrected, ActualTrial, average='weighted', zero_division=1)
    TrialRecall = metrics.precision_score(Trial_corrected, ActualTrial, average='weighted', zero_division=1)

    return TrialF1score, TrialPrecision, TrialRecall

def Evaluate_fold(SubjectScores, TestTrials, TestEpochs, scaler, Idx, FeatureSet, Clf, fold, pca):
    """Takes all the test trials of a fold and uses the trained classifier to predict them. 
    Then writes the scores of the fold on the SubjectScores structure"""   
    
    for trial in range(len(TestTrials)): # Run trough each trial
        
        Trial = np.array(TestTrials[trial]) # Select trial
        Epochs = np.array(TestEpochs[trial]) # Get actual onset and ending
        
        # Evaluate the selected trial, and return the metrics
        TrialF1score, TrialPrecision, TrialRecall = Evaluate_trial(Trial, Epochs, trial, scaler, Idx, FeatureSet, Clf, pca)   
        
        # Writes the obtained metrics of the trial
        SubjectScores['TrialF1scores'][trial,fold] = TrialF1score
        SubjectScores['TrialRecalls'][trial,fold] = TrialPrecision
        SubjectScores['TrialPrecisions'][trial,fold] = TrialRecall

    return SubjectScores





def Evaluate_subject(Results, subject, DS, FeatureSet, clf, Idx, script_dir):
    """Run the full evaluation on a subject. This is, trains a classifier with the train trials, and evaluates
    the classifier with the test trials. Does this process for each fold, and return the results obtained
    for that subject"""
    
    # get the folds of the subject
    SubjectFolds = Extract_subject_folds(subject, script_dir, DS)
    Size_train_trials = len(SubjectFolds[0]['TestTrials'])
    
    # Initializes structure that will record the obtained metrics on the evalauted test trials
    SubjectScores = {} # (#test_trials,#folds) = (Size_train_trials,4)
    SubjectScores['TrialF1scores'] = np.zeros((Size_train_trials,4))
    SubjectScores['TrialPrecisions'] = np.zeros((Size_train_trials,4))
    SubjectScores['TrialRecalls'] = np.zeros((Size_train_trials,4))
    
    for fold in range(len(SubjectFolds)): # Run trough each fold
        
        scaler = StandardScaler() # Creates the scaler to noramlize features
        Clf = Select_classifier(clf) # Creates the classifier
        
        Fold = SubjectFolds[fold] # Select fold
        
        # From the selected fold, gets the train and test trials, and actual onsets and endings of test trials
        TrainTrials = Fold['TrainTrials']
        TestTrials = Fold['TestTrials']
        TestEpochs = Fold['TestEpochs']
        
        BalancedCorpus = BalanceMultiTrials(TrainTrials) # Created a balanced corpus fromt he train trials
        ytrain = BalancedCorpus[:,-1] # Get targets
        
        Xtrain = BalancedCorpus[:,Idx] # Get train set (X)
        Xtrain = Remove_NaN(Xtrain) # To correct NaN and inf values
        
        scaler.fit(Xtrain) # Train the normalizer
        X = scaler.transform(Xtrain) # Normalizes the corpus
        

        
        # trains the classifier, if PCA is selected, applies PCA first
        if FeatureSet == 'PCA':
            pca = PCA(n_components=0.9, svd_solver = 'full')
            pca.fit(X)
            PCAtrain = pca.transform(X)
            Clf.fit(PCAtrain,ytrain)
        else:
            pca = 0
            Clf.fit(X,ytrain)

        # Evaluates the subject's fold and return the obtained scores
        SubjectScores = Evaluate_fold(SubjectScores, TestTrials, TestEpochs, scaler, Idx, FeatureSet, Clf, fold, pca)

    # Writes the obtained metrics of the subject on the Results structure
    Results['F1score'].append(SubjectScores['TrialF1scores'])
    Results['Recall'].append(SubjectScores['TrialRecalls'])
    Results['Precision'].append(SubjectScores['TrialPrecisions'])
    
    return Results
