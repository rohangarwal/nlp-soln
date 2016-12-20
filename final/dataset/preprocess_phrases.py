import re
import string
import pickle
from nltk.corpus import stopwords

stoplist = list(set(stopwords.words('english')))+['rt']
exclude = set(string.punctuation)

def filter_out(line):
    tmp = line.split(':')
    label = int(tmp[-1])
    line = ':'.join(tmp[:-1])
    #line = ' '.join([x for x in line.split() if x.decode('utf-8') not in stoplist])
    #line = ''.join([x.lower() for x in line if x not in exclude and not x.isdigit()])
    #line = ' '.join([x for x in line.split(' ') if len(x) > 1])
    return [line.strip(), label]

train_lines = []
with open("train_phrases.txt", "r") as f:
    lines = f.readlines()
    train_lines = [filter_out(line.strip()) for line in lines]
    train_lines = [x for x in train_lines if x[0]]
    
test_lines = []
with open("test_phrases.txt", "r") as f:
    lines = f.readlines()
    test_lines = [filter_out(line.strip()) for line in lines]
    test_lines = [x for x in test_lines if x[0]]
        
with open("pickles/train_phrases.pkl", "wb") as f:
    pickle.dump(train_lines, f)
    
with open("pickles/test_phrases.pkl", "wb") as f:
    pickle.dump(test_lines, f)
        
print('Train : ',len(train_lines), "Test :", len(test_lines))
