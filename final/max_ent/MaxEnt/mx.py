'''
Created on 23-Jul-2016

@author: Anantharaman
'''
from __future__ import division
import sys
import pickle
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from feature_function_tweets import WikiFeatureFunctions
from classifiers.maxent_base import LogLinear


ds_path = os.path.join(os.path.join(os.path.dirname(__file__)),"..", "data")

def get_wiki_file_names():
    fnames_1 = os.listdir(ds_path)
    fnames = [os.path.join(ds_path, f) for f in fnames_1]
    return fnames

def prepare_dataset(supported_labels):
    dataset = []
    fnames_1 = os.listdir(ds_path)
    fnames = [os.path.join(ds_path, f) for f in fnames_1]
    for fn, fn1 in zip(fnames, fnames_1):
        label = fn1 # our convention is that the filename will be name_label
        if label in supported_labels:
            txt = open(fn).read()
            # txt = txt[:100]
            dataset.append([txt, label])
    return dataset
    
def get_ans(txt):
    ff = WikiFeatureFunctions()
    supported_labels = ff.get_supported_labels()
    # print "Supported Labels: ", supported_labels
    dataset = prepare_dataset(supported_labels)
    clf = LogLinear(ff)
    clf.train(dataset, max_iter=50)
    result = {}
    tmp = {"0":0, "1":0, "2":0, "3":3, "4":4}
    for w in txt.split():
        result = clf.classify(w)[1]
        print(result)
        for key in result.keys():
            tmp[key] += result[key]
    
    key = max(tmp, key=tmp.get)
    return key

if __name__ == '__main__':
    ff = WikiFeatureFunctions()
    supported_labels = ff.get_supported_labels()
    # print "Supported Labels: ", supported_labels
    dataset = prepare_dataset(supported_labels)
    clf = LogLinear(ff)
    clf.train(dataset, max_iter=50)
    while True:
        txt = raw_input("Enter a text for classification: ")
        if txt == "__Q__":
            break
            
        result = {}
        tmp = {"0":0, "1":0, "2":0, "3":0, "4":0}
        for w in txt.split():
            result = clf.classify(w)[1]
            for key in result.keys():
                tmp[key] += result[key]
        
        key = max(tmp, key=tmp.get)
        print(key)
