import pickle
import numpy as np
from random import shuffle
import os

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
    res = str(np.argmax(ps))
    if res == '0':
    	return "compliment"
    elif res == '1':
    	return "displeasure"
    else:
    	return "miscellaneous"

if __name__ == "__main__":
    '''
    #test using word2vec
    posreviews = pickle.load(open('../word2vec/pos_vec_test.pkl',"rb"))
    negreviews = pickle.load(open('../word2vec/neg_vec_test.pkl',"rb"))
    '''
    dirname = '../pickles/files/'
    for fname in os.listdir(dirname):
	    datapkl = pickle.load(open(os.path.join(dirname, fname),"rb"))

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

	    shuffle(datapkl)
	    corr = 0
	    wrong = 0
	    num_comp, num_disp, num_misc = 0,0,0
	    for i in datapkl[:500]:
	    	res = test(i[0],hprev)
	    	if res == "compliment":
	    		num_comp += 1
	    	if res == "displeasure":
	    		num_disp += 1
	    	if res == "miscellaneous":
	    		num_misc += 1
	    	if res == i[1]:
	    		corr += 1
	    	else:
	    		wrong += 1
	    		
	    print "*"*10, fname, "*"*10
	    print "% of Displeasure :",float(num_disp)/len(datapkl[:500]), ", % of Misc :", float(num_misc)/len(datapkl[:500]), ", % of Compliment :", float(num_comp)/len(datapkl[:500])
	    print "Accuracy =", (float(corr)/(corr+wrong))*100
	    print "**************************************"
