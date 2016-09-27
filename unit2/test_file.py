import pickle
import numpy as np

def fwd(inputs,hprev):
  hsf[-1] = np.copy(hprev)
  for t in xrange(len(inputs)):
    xsf[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xsf[t][inputs[t]] = 1
    hsf[t] = np.tanh(np.dot(Wxhf, xsf[t]) + np.dot(Whhf, hsf[t-1]) + bhf) # hidden state
  return hsf

def bwd(inputs,hprev):
  hsb[len(inputs)] = np.copy(hprev)
  for t in reversed(xrange(len(inputs))):
    xsb[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xsb[t][inputs[t]] = 1
    hsb[t] = np.tanh(np.dot(Wxhb, xsb[t]) + np.dot(Whhb, hsb[t+1]) + bhb) # hidden state
  return hsb

def test(inputs,hprev,hpost):
  hf = fwd(inputs,hprev)
  hb = bwd(inputs,hpost)
  for t in range(len(inputs)):
    ys[t] = np.dot(Whyf, hf[t]) + np.dot(Whyb, hb[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    print str(np.argmax(ps[t])) + " ",

  print


if __name__ == '__main__':
	chars = [str(x) for x in range(32)]
	vocab_size = 32
	char_to_ix = { ch:i for i,ch in enumerate(chars) }
	xsf, hsf = {}, {}
	xsb, hsb = {}, {}
	ys, ps = {}, {}

	fp = open('model.pkl','rb')
	parameter_dict = pickle.load(fp)
	hprev = parameter_dict['hprev']
	hpost = parameter_dict['hpost']
	Whyf = parameter_dict['Whyf']
	Whyb = parameter_dict['Whyb']
	by = parameter_dict['by']
	Wxhf = parameter_dict['Wxhf']
	Whhf = parameter_dict['Whhf']
	bhf = parameter_dict['bhf']
	Wxhb = parameter_dict['Wxhb']
	Whhb = parameter_dict['Whhb']
	bhb = parameter_dict['bhb']
	fp.close()

	dt = open('sample.in', 'r').read().split("\n") # should be simple plain text file
	data = [x.split() for x in dt]
	for t in data:
		inputs = [char_to_ix[ch] for ch in t]
		test(inputs,hprev,hpost)


