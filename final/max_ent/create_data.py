import pytreebank
import re
import string
import pickle
from nltk.corpus import stopwords

stoplist = list(set(stopwords.words('english')))+['rt']
exclude = set(string.punctuation)

def filter_out(line):
    line = line.strip()
    line = ' '.join([x for x in line.split() if x not in stoplist])
    line = ''.join([x.lower() for x in line if x not in exclude and not x.isdigit()])
    return line.strip()

l0, l1, l2, l3, l4 = [], [], [], [], []
dataset = pytreebank.load_sst('trees')
for example in dataset['train']:
    for label,line in example.to_labeled_lines():
        line = filter_out(line)
        if line:
            if int(label) == 0:
                l0.extend(line.split(" "))
            if int(label) == 1:
                l1.extend(line.split(" "))
            if int(label) == 2:
                l2.extend(line.split(" "))
            if int(label) == 3:
                l3.extend(line.split(" "))
            if int(label) == 4:
                l4.extend(line.split(" "))

for example in dataset['test']:
    for label,line in example.to_labeled_lines():
        line = filter_out(line)
        if line:
            if int(label) == 0:
                l0.extend(line.split(" "))
            if int(label) == 1:
                l1.extend(line.split(" "))
            if int(label) == 2:
                l2.extend(line.split(" "))
            if int(label) == 3:
                l3.extend(line.split(" "))
            if int(label) == 4:
                l4.extend(line.split(" "))

tmp = l0 + l1 + l2 + l3 + l4
p0, p1, p2, p3, p4 = [], [], [], [], []
for x in set(tmp):
    a = [l0.count(x), l1.count(x), l2.count(x), l3.count(x), l4.count(x)]
    label = a.index(max(a))
    if max(a) > 2:
        if int(label) == 0:
            p0.append(x)
        if int(label) == 1:
            p1.append(x)
        if int(label) == 2:
            p2.append(x)
        if int(label) == 3:
            p3.append(x)
        if int(label) == 4:
            p4.append(x)
    else:
        p2.append(x)
    
with open("data/0", "w") as f:
    f.write('\n'.join(p0))

with open("data/1", "w") as f:
    f.write('\n'.join(p1))
    
with open("data/2", "w") as f:
    f.write('\n'.join(p2))
    
with open("data/3", "w") as f:
    f.write('\n'.join(p3))
    
with open("data/4", "w") as f:
    f.write('\n'.join(p4))
    
print("Dataset Created!", len(p0), len(p1), len(p2), len(p3), len(p4))
