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
    lines = [line.strip() for line in lines if len(line.split(",")) == 2 and "@" in list(line)]

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
compiled = re.compile(r'@\w+')
for line in lines:
    tags = re.findall(compiled, line)
    words = [w for w in line.split() if w not in tags]
    line = ' '.join(words)
    text = filter_out(line.split(",")[0]).strip()
    sentiment = line.split(",")[1]
    tweet = [text, sentiment]
    train.append(tweet)
    for tag in tags:
        if tag in tag_keys:
            name = tag_to_name[tag]
            if name in test.keys():
                test[name].append(tweet)
            else:
                test[name] = [tweet]
        
with open("pickles/train.pkl", "wb") as f:
    pickle.dump(train, f)
    
for k,v in test.items():
    with open('pickles/test_'+k+'.pkl', 'wb') as f:
        pickle.dump(v, f)
