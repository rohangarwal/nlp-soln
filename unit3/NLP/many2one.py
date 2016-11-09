import numpy as np
import pickle

# hyperparameters
hidden_size = 32 # size of hidden layer of neurons
learning_rate = 1e-1
vector_len = 32
outputs = 2

# model parameters
Wxh = np.random.randn(hidden_size, vector_len)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(outputs, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((outputs, 1)) # output bias

def lossFun(review, target, hprev):
  """
  inputs,target are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0

  # forward pass
  for t in range(len(review)):
    xs[t] = np.zeros((vector_len,1)) # encode in 1-of-k representation
    for j in range(32):
      xs[t][j] = review[t][j]
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state

  #Many 2 one
  last = len(review) - 1
  ys = np.dot(Why, hs[last]) + by # unnormalized log probabilities for next chars
  ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars
  loss = -np.log(ps[target,0]) # softmax (cross-entropy loss)

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  dy = np.subtract(ps,target) # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
  dWhy += np.dot(dy, hs[last].T)
  dby += dy
  dh = np.dot(Why.T, dy) + dhnext # backprop into h
  for t in reversed(range(len(review))):
    dhraw = (1 - (hs[t] * hs[t].T)) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[last]


if __name__ == '__main__':
  posreviews = pickle.load(open('../word2vec/pos_vec_train.pkl',"rb"))
  negreviews = pickle.load(open('../word2vec/neg_vec_train.pkl',"rb"))

  # Initializing model parameters
  mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  mbh, mby = np.zeros_like(bh), np.zeros_like(by)
  hprev = np.zeros((hidden_size,1))

  for review in posreviews:
    seq_length = len(review)
    smooth_loss = -np.log(1.0/vector_len)*seq_length # loss at iteration 0
    target = np.matrix('1;0')

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(review, target, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
      mem += dparam * dparam
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  for review in negreviews:
    seq_length = len(review)
    smooth_loss = -np.log(1.0/vector_len)*seq_length # loss at iteration 0
    target = np.matrix('0;1')

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(review, target, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
      mem += dparam * dparam
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  parameter_dict = {}
  parameter_dict['hprev'] = hprev
  parameter_dict['Why'] = Why
  parameter_dict['by'] = by
  parameter_dict['Wxh'] = Wxh
  parameter_dict['Whh'] = Whh
  parameter_dict['bh'] = bh

  fi = open("trained_model.pkl", "wb")
  pickle.dump(parameter_dict,fi)
  fi.close()



