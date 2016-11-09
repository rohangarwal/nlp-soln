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
    #print ps
    return str(np.argmax(ps))

if __name__ == "__main__":
    '''
    #test using word2vec
    posreviews = pickle.load(open('../word2vec/pos_vec_test.pkl',"rb"))
    negreviews = pickle.load(open('../word2vec/neg_vec_test.pkl',"rb"))
    '''
    #test using sir's implmn of word2vec
    posreviews = pickle.load(open('../sir2vec/pos_vec_test.pkl',"rb"))
    negreviews = pickle.load(open('../sir2vec/neg_vec_test.pkl',"rb"))

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

    TN = 0
    FN = 0
    TP = 0
    FP = 0
    for review in negreviews:
        if test(review,hprev) == str(0):
            FP += 1
        else:
            TN += 1

    for review in posreviews:
        if test(review,hprev) == str(0):
            TP += 1
        else:
            FN += 1

    #print "TP = " + str(TP) + "FN = " + str(FN) + "TN = " + str(TN) + "FP = " + str(FP)
    values = list()
    values = [TP,FN,TN,FP]
    with open('test_values.pkl', 'wb') as f:
        pickle.dump(values,f)
