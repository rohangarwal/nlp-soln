from __future__ import division
import pickle, sys
import numpy as np

hprev, Why, by = {}, {}, {}
Wxh, bh, Whh = {}, {}, {}

def fwd(phrase):
    global hprev, Why, by, Wxh, bh, Whh
    vector_length = 64
    xs = {}
    hs = {}
    hs[-1] = hprev
    for t in xrange(len(phrase)):
        xs[t] = np.zeros((vector_length,1))
        for j in range(vector_length):
            xs[t][j] = phrase[t][j]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    return hs[(len(phrase)-1)]

def test(phrase):
    global hprev, Why, by, Wxh, bh, Whh
    hs = fwd(phrase)
    ys = np.dot(Why, hs) + by
    ps = np.exp(ys) / np.sum(np.exp(ys))
    return str(np.argmax(ps))

def load(model):
    global hprev, Why, by, Wxh, bh, Whh
    fp = open(model,'rb')
    parameter_dict = {}
    parameter_dict = pickle.load(fp)
    hprev = parameter_dict['hprev']
    Why = parameter_dict['Why']
    by = parameter_dict['by']
    Wxh = parameter_dict['Wxh']
    Whh = parameter_dict['Whh']
    bh = parameter_dict['bh']
    fp.close()

def sentiment(sentence,model):
    load(model)
    return test(sentence)

def client(test_file,model):
    load(model)
    phrases = pickle.load(open(test_file,"rb"))
    Rite = 0
    m = {'0':['0','1'], '1':['2'], '2':['3','4']}
    for phrase in phrases:
        temp = str(test(phrase[0]))
        print temp
        if option == '3':
            if temp in m[temp]:
                Rite += 1
        else:
            if str(temp) == str(phrase[1]):
                Rite += 1
            
    print Rite
    print 'Accuracy - ', round(Rite/len(phrases),2)
    
if __name__ == '__main__':
    test_file, model, option = sys.argv[1:4]
    client(test_file, model)
