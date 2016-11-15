import re
import string
import pickle
from nltk.corpus import stopwords

stoplist = list(set(stopwords.words('english')))+['rt']
exclude = set(string.punctuation)

def filter_out(line):
    line = ' '.join([x for x in line.split() if x not in stoplist])
    line = ''.join([x.lower() for x in line if x not in exclude and not x.isdigit()])
    return line

lines = []
with open("FinalTweetList.csv", "r") as f:
    lines = f.readlines()
    sentiment = ['displeasure', 'compliment', 'miscellaneous']
    lines = [line.strip() for line in lines if len(line.split(",")) == 2 and "@" in list(line)]
    lines = [line for line in lines if line.split(",")[1].lower() in sentiment]

tag_to_name = {
        '@TheOfficialSBI':'SBI',
        '@HDFCBank_Cares':'HDFC',
        '@ICICIBank_Care':'ICICI',
        '@KotakBankLtd':'KOTAK',
        '@ICICIBank':'ICICI',
        '@RBI':'RBI',
        '@AxisBankSupport':'AXIS',
        '@HDFC_Bank':'HDFC',
        '@AxisBank':'AXIS',
        '@udaykotak':'KOTAK'
    }
    
tag_keys = tag_to_name.keys()

train = []
test = dict()
displesaure = []
compliment = []
misc = []
d_dis = dict()
d_comp = dict()
d_misc = dict()
compiled = re.compile(r'@\w+')
for line in lines:
    tags = re.findall(compiled, line)
    words = [w for w in line.split() if w not in tags]
    line = ' '.join(words)
    text = filter_out(line.split(",")[0]).strip()
    sentiment = line.split(",")[1]
    words = line.split()
    if sentiment == 'displeasure':
        for w in words:
            if w in d_dis.keys():
                d_dis[w] += 1
            else:
                d_dis[w] = 1            
    elif sentiment == 'compliment':
        for w in words:
            if w in d_comp.keys():
                d_comp[w] += 1
            else:
                d_comp[w] = 1
    else:    
        for w in words:
            if w in d_misc.keys():
                d_misc[w] += 1
            else:
                d_misc[w] = 1
                
for key in set(list(d_dis.keys())+list(d_comp.keys())+list(d_misc.keys())):
    c, d, m = 0, 0, 0
    if key in d_comp.keys():
        c = d_comp[key]
    if key in d_dis.keys():
        d = d_dis[key]
    if key in d_misc.keys():
        m = d_misc[key]
    
    key = ''.join([x.lower() for x in key if x not in exclude])    
    mx = max([c, m, d])
    if mx > 3 and key:
        if mx == c:
            for i in xrange(mx):
                compliment.append(key)
        elif mx == d:
            for i in xrange(mx):
                displesaure.append(key)
        else:
            for i in xrange(mx):
                misc.append(key)
       
with open("pickles/disp.txt", "w") as f:
    f.write("\n".join(displesaure))
with open("pickles/comp.txt", "w") as f:
    f.write("\n".join(compliment))
with open("pickles/misc.txt", "w") as f:
    f.write("\n".join(misc))
'''
c = compliment
d = displesaure
m = misc
final = []
for i in c:
    final.append([i, 'compliment'])
for i in d:
    final.append([i, 'displeasure'])
for i in m:
    final.append([i, 'miscellaneous'])
    
from random import shuffle
from pickle import dump
shuffle(final)
dump(final, open("pickles/new_pickle.pkl", "wb"))
'''
