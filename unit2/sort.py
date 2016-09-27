"""
BI-Directional RNN written by us...
"""
import pickle
import numpy as np

# data I/O
dt = open('dataset.in', 'r').read().split("\n") # should be simple plain text file
data = [x.split() for x in dt]
chars = [str(x) for x in range(32)]
vocab_size = 32
char_to_ix = { ch:i for i,ch in enumerate(chars) }
#ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 16 # size of hidden layer of neurons
learning_rate = 1e-1

# model parameters
Wxhf = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whhf = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Whyf = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bhf = np.zeros((hidden_size, 1)) # hidden bias

Wxhb = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whhb = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Whyb = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bhb = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
xsf, hsf = {}, {}
xsb, hsb = {}, {}
ys, ps = {}, {}

def fwd(inputs,hprev):
  hsf[-1] = np.copy(hprev)
  for t in xrange(len(inputs)):
    xsf[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xsf[t][inputs[t]] = 1
    hsf[t] = np.tanh(np.dot(Wxhf, xsf[t]) + np.dot(Whhf, hsf[t-1]) + bhf) # hidden state
  return hsf

def fwdback(hsf,ps):
  dWxh, dWhh, dWhy = np.zeros_like(Wxhf), np.zeros_like(Whhf), np.zeros_like(Whyf)
  dbh, dby = np.zeros_like(bhf), np.zeros_like(by)
  dhnext = np.zeros_like(hsf[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hsf[t].T)
    dby += dy
    dh = np.dot(Whyf.T, dy) + dhnext # backprop into h
    dhraw = (1 - hsf[t] * hsf[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xsf[t].T)
    dWhh += np.dot(dhraw, hsf[t-1].T)
    dhnext = np.dot(Whhf.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return dWxh, dWhh, dWhy, dbh, dby, hsf[len(inputs)-1]

def bwd(inputs,hprev):
  hsb[len(inputs)] = np.copy(hprev)
  for t in reversed(xrange(len(inputs))):
    xsb[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xsb[t][inputs[t]] = 1
    hsb[t] = np.tanh(np.dot(Wxhb, xsb[t]) + np.dot(Whhb, hsb[t+1]) + bhb) # hidden state
  return hsb

def bwdforward(hsb,ps):
  dWxh, dWhh, dWhy = np.zeros_like(Wxhb), np.zeros_like(Whhb), np.zeros_like(Whyb)
  dbh, dby = np.zeros_like(bhb), np.zeros_like(by)
  dhnext = np.zeros_like(hsb[0])
  for t in xrange(len(inputs)):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hsb[t].T)
    dby += dy
    dh = np.dot(Whyb.T, dy) + dhnext # backprop into h
    dhraw = (1 - hsb[t] * hsb[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xsb[t].T)
    dWhh += np.dot(dhraw, hsb[t+1].T)
    dhnext = np.dot(Whhb.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return dWxh, dWhh, dWhy, dbh, hsb[len(inputs)]

def train(inputs, targets, hprev, hpost):
  loss = 0
  # forward pass
  hf = fwd(inputs,hprev)
  hb = bwd(inputs,hpost)

  for t in range(len(inputs)):
    ys[t] = np.dot(Whyf, hf[t]) + np.dot(Whyb, hb[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars

    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

  dWxhf, dWhhf, dWhyf, dbhf, dby, hprev = fwdback(hf,ps)
  dWxhb, dWhhb, dWhyb, dbhb, hpost = bwdforward(hb,ps)
  return loss, dWxhf, dWhhf, dWhyf, dbhf, dby, hprev, dWxhb, dWhhb, dWhyb, dbhb, hpost
  # backward pass: compute gradients going backwards

mWxhf, mWhhf, mWhyf = np.zeros_like(Wxhf), np.zeros_like(Whhf), np.zeros_like(Whyf)
mbhf, mby = np.zeros_like(bhf), np.zeros_like(by) # memory variables for Adagrad

mWxhb, mWhhb, mWhyb = np.zeros_like(Wxhb), np.zeros_like(Whhb), np.zeros_like(Whyb)
mbhb = np.zeros_like(bhb) # memory variables for Adagrad
#smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
hprev = np.zeros((hidden_size,1))
hpost = np.zeros((hidden_size,1))
for n,mn in enumerate(data):

  inputs = [char_to_ix[ch] for ch in mn]
  targets = [char_to_ix[ch] for ch in sorted(mn, key = lambda x : int(x))]

  # prepare inputs (we're sweeping from left to right in steps seq_length long)

  # forward seq_length characters through the net and fetch gradient
  loss, dWxhf, dWhhf, dWhyf, dbhf, dby, hprev, dWxhb, dWhhb, dWhyb, dbhb, hpost = train(inputs, targets, hprev, hpost)

  #smooth_loss = smooth_loss * 0.999 + loss * 0.001
  #if n % 2 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxhf, Whhf, Whyf, bhf, Wxhb, Whhb, Whyb, bhb, by],
                                [dWxhf, dWhhf, dWhyf, dbhf, dWxhb, dWhhb, dWhyb, dbhb, dby],
                                [mWxhf, mWhhf, mWhyf, mbhf, mWxhb, mWhhb, mWhyb, mbhb, mby]
                                ):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

parameter_dict = {}
parameter_dict['hprev'] = hprev
parameter_dict['hpost'] = hpost
parameter_dict['Whyf'] = Whyf
parameter_dict['Whyb'] = Whyb
parameter_dict['by'] = by
parameter_dict['Wxhf'] = Wxhf
parameter_dict['Whhf'] = Whhf
parameter_dict['bhf'] = bhf
parameter_dict['Wxhb'] = Wxhb
parameter_dict['Whhb'] = Whhb
parameter_dict['bhb'] = bhb

fi = open("model.pkl", "wb")
pickle.dump(parameter_dict,fi)
fi.close()