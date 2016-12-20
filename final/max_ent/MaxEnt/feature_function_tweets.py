'''
Created on 23-Jul-2016
@author: Anantharaman Narayana Iyer

Conventions for writing feature functions:
1. first letter of function name should be f, followed by the label, underscore, function number
e.g. fsports_1(x, y)

'''
import sys, os
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__)))
from nltk import sent_tokenize, word_tokenize
from classifiers.feature_functions_base import FeatureFunctionsBase

class WikiFeatureFunctions(FeatureFunctionsBase):
    def __init__(self):
        super(WikiFeatureFunctions, self).__init__()
        self.fdict = {}
        self.f0 = []
        self.f1 = []
        self.f2 = []
        self.f3 = []
        self.f4 = []
        for k, v in WikiFeatureFunctions.__dict__.items():
            if hasattr(v, "__call__"):
                if k[0] == 'f':
                    tag = k[1:].split("_")[0]
                    val = self.fdict.get(tag, [])
                    val.append(v)
                    self.fdict[tag] = val
        self.supported_tags = self.fdict.keys()
        feature_vec = pickle.load(open(os.path.join(os.path.dirname(__file__))+"/../pickles/feature.pkl" , "rb"))
        for i,j in feature_vec.items():
            if i == "f0":
                self.f0 = list(j)
            elif i == "f1":
                self.f1 = list(j)
            elif i == "f2":
                self.f2 = list(j)
            elif i == "f3":
                self.f3 = list(j)
            elif i == "f4":
                self.f4 = list(j)
        return

    def check_membership(self, ref_set, my_set):
        """Check if there is any non null intersection between 2 sets"""
        rset = set(ref_set)
        mset = set(my_set)
        if len(rset.intersection(mset)) > 0:
            return 1
        else:
            return 0

    # you may write as many functions as you require
    # you should return 0 or 1 for each feature function
    # words is a word tokenized text document and y is a label


    def f0_1(self, words, y):
        if (self.check_membership(self.f0, words)) and (y == "0"):
            return 1
        return 0

    def f1_1(self, words, y):
        if (self.check_membership(self.f1, words)) and (y == "1"):
            return 1
        return 0

    def f2_1(self, words, y):
        if (self.check_membership(self.f2, words)) and (y == "2"):
            return 1
        return 0
        
    def f3_1(self, words, y):
        if (self.check_membership(self.f3, words)) and (y == "3"):
            return 1
        return 0
        
    def f4_1(self, words, y):
        if (self.check_membership(self.f4, words)) and (y == "4"):
            return 1
        return 0

    def evaluate(self, x, y):
        words = []
        stoks = sent_tokenize(x.lower())
        # create a linear array of word tokens
        for tok in stoks:
            words.extend(word_tokenize(tok))
        return FeatureFunctionsBase.evaluate(self, words, y)



if __name__ == "__main__":
    ff = WikiFeatureFunctions()
    print ff.get_supported_labels()
