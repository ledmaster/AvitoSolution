# -*- coding: utf-8 -*-
import csv
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.grid_search import GridSearchCV
import numpy as np
import time
#import pandas as pd
import nltk.corpus
from sklearn.linear_model import SGDClassifier, Lasso
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from scipy.sparse import csr_matrix,hstack,bmat
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
import pickle
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from multiprocessing import Pool
from avitoUtils import *
import re


stopwords= [word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian")]

def Classifier(filename):
    
    

    print 'Loading data...'
    id, data, target = readTrainData(filename)
    
    
    print 'Total Examples',data.shape[0], 'Dummy percentage',1 - target.mean()
    
    
    accuracy = []
    
    kf = cross_validation.StratifiedKFold(target,5)
    
    print 'Training and Testing...'
    
    for train, test in kf:
        dataTrain, dataTest, targetTrain, targetTest = data[train], data[test], target[train], target[test]
        
        idTest = id[test]
        
            
        clf = BlendedClassifiers()
        
        clf.fit(dataTrain,targetTrain)
        
        probs = clf.predict_proba(dataTest)

        metric = PAtK( probs, targetTest, idTest)
        
        accuracy.append(metric)
        
        # print clf.predict_proba(dataTest)
        print 'P@K:', metric
    
    mean = np.mean(accuracy)
    ci = 1.96*(np.std(accuracy)/np.sqrt(5))

    print 'Mean P@K', mean, 'CI  95%', mean - ci,'-',mean+ci

    return accuracy


def clfSub():
    
    data_files = os.listdir(os.getcwd())
    
    for file in data_files:
        if 'tsv' not in file or 'stemTest' in file:
            continue


        print file
        id,data, target = readTrainData(file)

        clf = BlendedClassifiers()

        clf.fit(data,target)
    
        Classifier = open(file+'.clf','w+')

        pickle.dump(clf,Classifier)
        Classifier.close()


def testMultipleFiles():

    files = ['Животные.tsv','Для бизнеса.tsv']
    #files = os.listdir(os.getcwd())
    accuracy = []

    for file in files:
        if '.tsv' in file and 'stemTest' not in file:
            print file
            accuracy = accuracy + Classifier(file)


    mean = np.mean(accuracy)
    ci = 1.96*(np.std(accuracy)/np.sqrt(len(accuracy)))

    print 'Total P@K', mean, '95% CI', mean - ci,'-',mean+ci


    
    



class BlendedClassifiers(object):

    def __init__(self):
        self.vectorizer = TfidfVectorizer(binary=False,min_df=1,strip_accents='unicode',stop_words=stopwords)
        #self.selector = SelectKBest(k=12000,score_func=chi2)
        self.selector = VarianceThreshold()
        self.clf1 = MultinomialNB(alpha=1e-1)
        self.clf2 = SGDClassifier(n_jobs=3,loss='modified_huber',shuffle=True,warm_start=True, alpha=1e-5)
    
        
    def fit(self,data,target):


        data = self.vectorizer.fit_transform(data)
        data = self.selector.fit_transform(data,target)
        


        self.clf1.fit(data,target)
        self.clf2.fit(data,target)
    
    def predict_proba(self,data,inside=False):
        
        data = self.vectorizer.transform(data)
        data = self.selector.transform(data)
        
        probs = np.column_stack([self.clf1.predict_proba(data).T[1],
                                 self.clf2.predict_proba(data).T[1]]).mean(axis=1)
        
        return probs















#subsetDirFiles()
#Classifier()
#cleanDirFiles()
#cleanWordsTest()
#clfSub()
createSub(test=False)
#testMultipleFiles()
#plotData()






