import pickle
import numpy as np

def fwd(review,hprev):
    vector_length = 32
    xs = {}
    hs = {}
    hs[-1] = hprev
    for t in xrange(len(review)):
        xs[t] = np.zeros((vector_length,1))
        for j in range(32):
            xs[t][j] = review[t][j]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    return hs[(len(review)-1)]

def test(review,hprev):
    hs = fwd(review,hprev)
    ys = np.dot(Why, hs) + by
    ps = np.exp(ys) / np.sum(np.exp(ys))
    print ps
    return str(np.argmax(ps))

if __name__ == "__main__":
    posreviews = pickle.load(open('../word2vec/pos_vec_test.pkl',"rb"))
    negreviews = pickle.load(open('../word2vec/neg_vec_test.pkl',"rb"))

    parameter_dict = {}
    fp = open('trained_model.pkl','rb')
    parameter_dict = pickle.load(fp)
    hprev = parameter_dict['hprev']
    Why = parameter_dict['Why']
    by = parameter_dict['by']
    Wxh = parameter_dict['Wxh']
    Whh = parameter_dict['Whh']
    bh = parameter_dict['bh']
    fp.close()
    for review in negreviews:
        print test(review,hprev)
