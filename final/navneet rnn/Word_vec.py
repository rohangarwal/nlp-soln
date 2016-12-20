import random
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec
import cPickle

tknzr = TweetTokenizer()
sentence = []


train_sentence = []
train_sen_c = []
train_result = []

test_sentence = []
test_sen_c = []
test_result = []


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
for i in dumpx:
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
		t1 = len(l3)
		sentence.append(res)
		if(st<lenx1):
			train_sentence.append(res)
			train_sen_c.append(t1)
			train_result.append(ret_d)

		else:
			test_sentence.append(res)
			test_sen_c.append(t1)
			test_result.append(ret_d)
	except Exception as e:
		pass

'''
print len(train_sentence)
print len(train_sen_c)
print len(train_result)

print len(test_sentence)
print len(test_sen_c)
print len(test_result)

'''
print "now training"
model = Word2Vec(sentence, min_count=min_count, size=size)
print "now processing"
cPickle.dump(model, open('save1.p', 'wb')) 
cPickle.dump(train_sentence, open('save2.p', 'wb')) 
cPickle.dump(train_result, open('save3.p', 'wb')) 
cPickle.dump(train_sen_c, open('save4.p', 'wb')) 
cPickle.dump(test_sentence, open('save5.p', 'wb')) 
cPickle.dump(test_result, open('save6.p', 'wb')) 
cPickle.dump(test_sen_c, open('save7.p', 'wb')) 

'''
for s1 in sentence:
	lx = []
	for w1 in s1:
		temp = model[w1]
		lx.append(temp)
	vec.append(lx)
'''
