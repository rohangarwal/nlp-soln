def fwd(review):
  for t in xrange(len(review)):
  	xs[t] = review[t]
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    '''
    xsf[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xsf[t][inputs[t]] = 1
    hsf[t] = np.tanh(np.dot(Wxhf, xsf[t]) + np.dot(Whhf, hsf[t-1]) + bhf) # hidden state
  	'''
  return hs

def test(review):
	hf = fwd(review,hprev)
	temp = []
  	for t in range(len(review)):
    	ys[t] = np.dot(Whyf, hf[t]) + np.dot(Whyb, hb[t]) + by # unnormalized log probabilities for next chars
    	ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    	temp.append(str(np.argmax(ps[t])))

  return ' '.join(temp)

if __name__ == "__main__":

	fp = open('trained_model.pkl','rb')
		parameter_dict = pickle.load(fp)
		hprev = parameter_dict['hprev']
		Why = parameter_dict['Why']
		by = parameter_dict['by']
		Wxh = parameter_dict['Wxh']
		Whh = parameter_dict['Whh']
		bh = parameter_dict['bh']
	fp.close()
