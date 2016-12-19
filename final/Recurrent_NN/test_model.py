import pickle, sys
import numpy as np
from __future__ import division

def fwd(phrase,hprev):
    vector_length = 32
    xs = {}
    hs = {}
    hs[-1] = hprev
    for t in xrange(len(phrase)):
        xs[t] = np.zeros((vector_length,1))
        for j in range(32):
            xs[t][j] = phrase[t][j]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    return hs[(len(phrase)-1)]

def test(phrase,hprev):
    hs = fwd(phrase,hprev)
    ys = np.dot(Why, hs) + by
    ps = np.exp(ys) / np.sum(np.exp(ys))
    return str(np.argmax(ps))

if __name__ == "__main__":
    phrases = pickle.load(open(sys.argv[-1],"rb")) #Enter testing file

    parameter_dict = {}
    fp = open(sys.argv[0],'rb') # Enter model to be loaded
    parameter_dict = pickle.load(fp)
    hprev = parameter_dict['hprev']
    Why = parameter_dict['Why']
    by = parameter_dict['by']
    Wxh = parameter_dict['Wxh']
    Whh = parameter_dict['Whh']
    bh = parameter_dict['bh']
    fp.close()

    Rite = 0
    for phrase in phrases:
        if test(phrase[0],hprev) == phrase[1]:
            Rite += 1

    print 'Accuracy - ' round(Rite/len(phrases),2)
