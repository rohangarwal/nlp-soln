import operator
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]

onlyfiles = [x for x in onlyfiles if 'companies' in x]

h = {}

for file_ in onlyfiles:
    tmp = open(file_, 'r').read().strip()
    for i in tmp.split(' '):
        if i.lower() in h:
            h[i.lower()] += 1
        else:
            h[i.lower()] = 1


from nltk.corpus import stopwords
stop = set(stopwords.words('english'))


for k,v in sorted(h.items(), key=operator.itemgetter(1)):
    if k not in stop:
        print(k,' : ', v)
