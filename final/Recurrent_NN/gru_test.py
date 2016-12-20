from __future__ import division
import pickle, sys
import numpy as np

hprev, Why, Wc, Wr = {}, {}, {}, {}
Wz, Uc, Ur, Uz = {}, {}, {}, {}
by, bc, br, bz = {}, {}, {}, {}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fwd(phrase):
    global hprev, Why, Wc, Wr, Wz, Uc, Ur, Uz, by, bc, br, bz
    vector_len = 32
    xs, hs, ys, ps = {}, {}, {}, {}
    rs, zs, cs = {}, {}, {}
    rbars, zbars, cbars = {}, {}, {}
    hs[-1] = np.copy(hprev)

    for t in range(len(phrase)):
        xs[t] = np.zeros((vector_len,1)) # encode in 1-of-k representation
        #Copying entire vector for each word

        for j in range(32):
            xs[t][j] = phrase[t][j]

        #GRU Implementation
        rbars[t] = np.dot(Wr, xs[t]) + np.dot(Ur, hs[t-1]) + br
        rs[t] = sigmoid(rbars[t])

        # The z gate, which interpolates between candidate and h[t-1] to compute h[t]
        zbars[t] = np.dot(Wz, xs[t]) + np.dot(Uz, hs[t-1]) + bz
        zs[t] = sigmoid(zbars[t])

        # The candidate, which is computed and used as described above.
        cbars[t] = np.dot(Wc, xs[t]) + np.dot(Uc, np.multiply(rs[t] , hs[t-1])) + bc
        cs[t] = np.tanh(cbars[t])

        ones = np.ones_like(zs[t])
        hs[t] = np.multiply(cs[t],zs[t]) + np.multiply(hs[t-1],ones - zs[t])

    last = len(phrase) - 1
    return hs[last]

def test(phrase):
    global hprev, Why, Wc, Wr, Wz, Uc, Ur, Uz, by, bc, br, bz
    hs = fwd(phrase)
    ys = np.dot(Why, hs) + by
    ps = np.exp(ys) / np.sum(np.exp(ys))
    return np.argmax(ps)

def load(model):
    global hprev, Why, Wc, Wr, Wz, Uc, Ur, Uz, by, bc, br, bz
    fp = open(model,'rb')

    parameter_dict = {}
    parameter_dict = pickle.load(fp)

    # Loading all parameters
    hprev = parameter_dict['hprev']
    Why = parameter_dict['Why']
    Wc = parameter_dict['Wc']
    Wr = parameter_dict['Wr']
    Wz = parameter_dict['Wz']
    Uc = parameter_dict['Uc']
    Ur = parameter_dict['Ur']
    Uz = parameter_dict['Uz']
    by = parameter_dict['by']
    bc = parameter_dict['bc']
    br = parameter_dict['br']
    bz = parameter_dict['bz']
    fp.close()

def sentiment(sentence,model):
    load(model)
    return test(sentence)

def client(test_file,model):
    load(model)
    phrases = pickle.load(open(test_file,"rb"))
    Rite = 0
    for phrase in phrases:
        temp = test(phrase[0])
        print temp
        if temp == str(phrase[1]):
            Rite += 1
    print Rite
    print 'Accuracy - ', round(Rite/len(phrases),2)
