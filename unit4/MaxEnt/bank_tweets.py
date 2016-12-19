'''
Created on 23-Jul-2016

@author: Anantharaman
'''
from __future__ import division
import sys
import pickle
sys.path.append('/home/Work/Git/nlp-soln/unit4/MaxEnt')
from feature_function_tweets import WikiFeatureFunctions
from classifiers.maxent_base import LogLinear
import os

ds_path = os.path.join("..","data")

def get_wiki_file_names():
    fnames_1 = os.listdir(ds_path)
    fnames = [os.path.join(ds_path, f) for f in fnames_1]
    return fnames

def prepare_dataset(supported_labels):
    dataset = []
    fnames_1 = os.listdir(ds_path)
    fnames = [os.path.join(ds_path, f) for f in fnames_1]
    for fn, fn1 in zip(fnames, fnames_1):
        label = fn1.split("_")[1] # our convention is that the filename will be name_label
        if label in supported_labels:
            txt = open(fn).read()
            txt = txt[:20]
            dataset.append([txt, label])
    return dataset

if __name__ == '__main__':
    ff = WikiFeatureFunctions()
    supported_labels = ff.get_supported_labels()
    print "Supported Labels: ", supported_labels
    dataset = prepare_dataset(supported_labels)
    clf = LogLinear(ff)
    clf.train(dataset, max_iter=50)

    #Testing
    file = open('../pickles/test_SBI.pkl','rb')
    dic = pickle.load(file)
    correct = 0
    displease = 0
    total = len(dic)

    for ls in dic:
        '''
        txt = raw_input("Enter a text for classification: ")
        if txt == "__Q__":
            break
        '''
        tmp = {"compliment":0, "displeasure":0, "miscellaneous":0}
        for w in ls[0].split():
            result = clf.classify(w)[1]
            for key in result.keys():
                tmp[key] += result[key]

        key = max(tmp, key=tmp.get)
        if key == ls[1]:
            correct += 1
        if key == 'displeasure':
            displease += 1
    print 'Displeasure - ',(displease/total)*100
    print 'Accuracy - ',(correct/total)*100
