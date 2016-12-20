'''
Created on 23-Jul-2016
@author: Anantharaman Narayana Iyer

Conventions for writing feature functions:
1. first letter of function name should be f, followed by the label, underscore, function number
e.g. fsports_1(x, y)

'''
import sys
import pickle
sys.path.append('/home/sai/Desktop/nlp/nlp-soln/unit4')
from nltk import sent_tokenize, word_tokenize
from classifiers.feature_functions_base import FeatureFunctionsBase

class WikiFeatureFunctions(FeatureFunctionsBase):
    def __init__(self):
        super(WikiFeatureFunctions, self).__init__()
        self.fdict = {}
        self.disp = []
        self.comp = []
        self.misc = []
        for k, v in WikiFeatureFunctions.__dict__.items():
            if hasattr(v, "__call__"):
                if k[0] == 'f':
                    tag = k[1:].split("_")[0]
                    val = self.fdict.get(tag, [])
                    val.append(v)
                    self.fdict[tag] = val
        self.supported_tags = self.fdict.keys()  
        feature_vec = pickle.load(open("../pickles/featurewords.pkl" , "rb"))
        for i,j in feature_vec.items():
            if i == "displeasure":
                self.disp = list(j)
            elif i == "compliment"    :
                self.comp = list(j)

            else :
                self.misc = list(j) 
        print self.disp
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
    
 
    def fcompliment_1(self, words, y):
        comp = self.comp
        if (self.check_membership(comp, words)) and (y == "compliment"):
            return 1
        return 0
        
    def fdispleasure_1(self, words, y):
        disp = self.disp
        if (self.check_membership(disp, words)) and (y == "displeasure"):
            return 1
        return 0
        
    def fmiscellaneous_1(self, words, y):
        misc = self.misc
        if (self.check_membership(misc, words)) and (y == "miscellaneous"):
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
  