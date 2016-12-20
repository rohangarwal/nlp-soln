import random
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec
import cPickle
import requests
import math
import requests
import json
import random
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec

tknzr = TweetTokenizer()
sentence = []



serviceURL = "http://www.jnresearchlabs.com:9027/" # NOTE: we will get rid of port number later!
runCommandURL = serviceURL + "run_command" # this is the end point to which we will POST the command and params
headers = {'content-type': 'application/json'}

def get_vec_for_words(words): 
    """Given a list of words return the corresponding word vectors - default dimensions = 32"""
    r = requests.post(runCommandURL, data = json.dumps({"cmd": "word_reps", "params": {"txt": words}}), headers = headers) #
    return json.loads(r.text)

def test_service(inp):
    ret = []
    result = get_vec_for_words(inp.split())

    for i in result:
        if(i == "reps"):
            ret = result[i]
    return ret


train_sentence = []
train_sen_c = []
train_result = []

test_sentence = []
test_sen_c = []
test_result = []

vec= []
vecr = []

min_count = 0
size = 32

dumpx = []
f1 = open("neg1.txt");
for i in f1:
	dumpx.append(i)

f1 = open("pos1.txt");
for i in f1:
	dumpx.append(i)

random.shuffle(dumpx)

lenx1 = float(27443*85) /100
lenx1 = int(lenx1)
print lenx1

st = 0

count = 1000
for i in dumpx:
	print count
	st = st+1
	l1 = i.split("~^")
	try:
		ret_d = int(float(l1[1]))
		if(ret_d == 5):
			ret_d = 1
		else:
			ret_d = 0
		l2 = tknzr.tokenize(l1[0])
		l3 = tknzr.tokenize(l1[2])
		res = l2+l3
		ret = ""
		for i in res:
			ret = ret + i+" ";
		vec.append(test_service(str(ret)))
		vecr.append(ret_d)
		count = count-1
		if(count < 1):
			break
	except Exception as e:
		pass


cPickle.dump(vec, open('save8.p', 'wb')) 
cPickle.dump(vecr, open('save9.p', 'wb')) 

