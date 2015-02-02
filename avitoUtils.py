# -*- coding: utf-8 -*-
import csv
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import nltk.corpus
import nltk
import re
import string
import os
import pylab
from multiprocessing import Pool
import pickle

def PAtK(probs,target, id):
    
    sorted_ids = sorted(zip(probs,id), reverse = True)
    
    id_score = {id[i]: target[i] for i in range(id.shape[0])}
    
    percent_best = int(len(target)*min(np.mean(target),0.05))
    #percent_best = int(len(target)*np.mean(target))
    
    correct = 0.
    
    print 'Taking the',percent_best,'best.'
    for prob, id in sorted_ids[:percent_best]:
        if id_score[id] == 1:
            correct += 1
    
    return correct/percent_best


def createSub(test=False):

    data_files = os.listdir(os.getcwd())

    classifiers = {}
    data = {}

    print 'Loading classifiers...'

    for file in data_files:
        if '.clf' in file:
            
            print file
            
            class_file = open(file,'r')
            classifiers[file[:-8]] = pickle.load(class_file)
            class_file.close()

    file = open('stemTest.tsv','r')
    rdr = csv.reader(file,delimiter='\t',quotechar='"')

    ctr = 0

    for line in rdr:
        ctr += 1
        id = int(line[0])
        if line[1] in data.keys():
            data[line[1]].append((id,line[2]))
        else:
            data[line[1]] = [(id, line[2])]
        
        if ctr % 50000 == 0:
            if test:
                break
            print ctr



    for cat in data.keys():
        print 'Predicting ', cat
        data[cat] = np.array(data[cat])
        data[cat] = np.column_stack([classifiers[cat].predict_proba(data[cat][:,1]),data[cat][:,0]])

    cats = data.keys()
    final_data = data[cats[0]]

    for cat in cats[1:]:
        print 'Concatenating', cat
        final_data = np.concatenate([final_data, data[cat]])

    print 'Transforming types'
    final_data = final_data.astype(np.float)

    final_data = final_data[final_data[:,0].argsort()]
    final_data = final_data[::-1]

    print final_data

    print 'Writing solution..'

    f = open('avito_solution.csv','w+')
    f.write('id\n')
    for item in final_data:
        f.write("%d\n" % (int(item[1])))
    f.close()



def subsetData(filename, falseDataMultiplier=4):
    
    id,data,target = readTrainData(filename)
    
    proportion = 1 - np.mean(target)
    print filename, proportion
    if proportion < 0.9:
        return id,data,target
    
    true_data = data[target==True]
    true_target = target[target==True]
    true_id = id[target==True]
    
    false_data = data[target==False]
    false_target = target[target==False]
    false_id = id[target==False]
    
    false_idx = np.random.choice(np.arange(false_data.shape[0]),true_data.shape[0]*falseDataMultiplier)
    false_data = false_data[false_idx]
    false_target = false_target[false_idx]
    false_id = false_id[false_idx]


    id = np.concatenate([true_id,false_id])
    data = np.concatenate([true_data,false_data])
    target = np.concatenate([true_target, false_target])
    
    
    np.savetxt(filename, np.column_stack([id,data,target]), delimiter='\t', fmt='%s\t%s\t%s')

    return id,data,target








def readTrainData(filename):
    """ Junta todos os textos numa string, para vetorizar"""
    file = open(filename,'r')
    rdr = csv.reader(file,delimiter='\t')
    ctr = 0
    
    id = []
    texts = []
    target = []
    
    
    for line in rdr:
        
        id.append(int(line[0]))
        texts.append(line[1])
        target.append(int(line[2]))
    
    id = np.array(id)
    texts = np.array(texts)
    target = np.array(target)
    
    return id,texts, target

def writeToCatFile(dataDic):
    """Verifica se existe arquivo para aquela categoria, se nao houver, cria
        Registra aquela data naquela categoria
        """
    
    for key in dataDic.keys():
        filename = key + '.tsv'
        data_files = os.listdir(os.getcwd())
        if filename not in data_files:
            file = open(filename, 'w+')
        else:
            file = open(filename,'a+')
        
        wrt = csv.writer(file,delimiter='\t')
        wrt.writerows(dataDic[key])
        file.close()



def sepDataByCat():
    """Itera pelas linhas do training file
        Verifica a categoria e registra-o no arquivo da mesma
        
        """
    file = open('/Users/mnestevao/ODesk/Kaggle/Avito/avito_train.tsv','r')
    rdr = csv.DictReader(file,delimiter='\t',quotechar='"')
    
    newData = {}
    
    ctr = 0
    
    for line in rdr:
        ctr += 1
        data = [line['itemid'],line['subcategory'] + ' ' + line['title'] + ' ' + line['description'], line['is_blocked']]
        
        cat = line['category']
        
        if cat not in newData.keys():
            newData[cat] = [data]
        else:
            newData[cat].append(data)
        
        if ctr % 100000 == 0:
            writeToCatFile(newData)
            newData = {}
            print ctr

def readTestData(filename):
    """ Junta todos os textos numa string, para vetorizar"""
    file = open(filename,'r')
    rdr = csv.reader(file,delimiter='\t')
    ctr = 0
    
    texts = []
    
    
    for line in rdr:
        
        texts.append(line[2])
    
    
    texts = np.array(texts)
    
    return texts


def checkWordFreq():
    
    stopwords= frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian"))

    id,data,target = readTrainData('\xd0\x96\xd0\xb8\xd0\xb2\xd0\xbe\xd1\x82\xd0\xbd\xd1\x8b\xd0\xb5.tsv')

    vect = CountVectorizer(binary=False,min_df=1,strip_accents='unicode',max_features=1000)

    data = vect.fit_transform(data)
    
    """
    for mean, word in sorted(zip(list(data.mean(axis=0).T),vect.get_feature_names()),reverse=True):
        #print word, mean[0][0]
        print word"""

    for word in sorted(vect.get_feature_names()):
        print word


def plotData():
    
    stopwords= frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian"))
    
    id,data,target = readTrainData('Животные.tsv')
    
    vect = TfidfVectorizer(binary=False,min_df=1,strip_accents='unicode',max_features=10000000)
    svd = TruncatedSVD(n_components=2)
    
    data = vect.fit_transform(data)
    data = svd.fit_transform(data)
    
#print svd.explained_variance_ratio_.sum()
    
    pylab.scatter(data[:,0], data[:,1],c=target)
    pylab.show()


def cleanIndiv(line):

    
    
    ## clean numbers
    newData = re.sub('[0-9]+', '',line[1])
    ## clean punctuation
    newData = newData.translate(string.maketrans("",""), string.punctuation)
    

    
    return [line[0], newData, int(line[2])]

def cleanIndivTest(line):
    
    ## clean numbers
    newData = re.sub('[0-9]+', '',line[2])
    ## clean punctuation
    newData = newData.translate(string.maketrans("",""), string.punctuation)
    
    
    return [line[0],line[1],newData]

def cleanWords(filename):
    
    file = open(filename,'r')
    rdr = csv.reader(file,delimiter='\t')
    
    p = Pool(3)
    
    data = p.map(cleanIndiv,rdr)
    
    p.terminate()

    file.close()

    file = open(filename,'w')
    wrt = csv.writer(file,delimiter='\t')

    wrt.writerows(data)

    file.close()


def cleanWordsTest(filename='stemTest.tsv'):
    
    file = open(filename,'r')
    rdr = csv.reader(file,delimiter='\t')
    
    p = Pool(3)
    
    data = p.map(cleanIndivTest,rdr)
    
    p.terminate()
    
    file.close()
    
    file = open(filename,'w')
    wrt = csv.writer(file,delimiter='\t')
    
    wrt.writerows(data)
    
    file.close()


def cleanDirFiles():
    for file in os.listdir(os.getcwd()):
        if '.tsv' in file and 'stemTest' not in file:
            print file
            cleanWords(file)


def subsetDirFiles():
    for file in os.listdir(os.getcwd()):
        if '.tsv' in file and 'stemTest' not in file:
            print file
            subsetData(file)




#checkWordFreq()
