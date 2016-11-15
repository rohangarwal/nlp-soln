import re
import string
import pickle
from nltk.corpus import stopwords

stoplist = set(stopwords.words('english'))
exclude = set(string.punctuation)

def filter_out(line):
    line = ' '.join([x for x in line.split() if x not in stoplist])
    line = ''.join([x.lower() for x in line if x not in exclude and not x.isdigit()])
    return line

lines = []
with open("FinalTweetList.csv", "r") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines if len(line.split(",")) == 2 and "@" in list(line)]

result = []
compiled = re.compile(r'@\w+')
for line in lines:
    tags = re.findall(compiled, line)
    words = [w for w in line.split() if w not in tags]
    line = ' '.join(words)
    text = filter_out(line.split(",")[0])
    sentiment = line.split(",")[1]
    for tag in tags:
        result.append([tag, text, sentiment])
        
with open("dataset.bin", "wb") as f:
    pickle.dump(result, f)
