import numpy as np
import pickle

# hyperparameters
hidden_size = 32 # size of hidden layer of neurons
learning_rate = 1e-1
vector_len = 32
outputs = 5 #No of dimensions of output

# model parameters
Wxh = np.random.randn(hidden_size, vector_len)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(outputs, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((outputs, 1)) # output bias

def lossFun(phrase, target, hprev):
  """
  inputs,target are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0

  # forward pass
  # xs represents entire phrase/sentence
  for t in range(len(phrase)):
    xs[t] = np.zeros((vector_len,1)) # encode in 1-of-k representation
    #Copying entire vector for each word
    for j in range(32):
      xs[t][j] = phrase[t][j] #Using tanh, **Modify Here**
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state

  #Many 2 one
  last = len(phrase) - 1  # Getting only last hidden state
  ys = np.dot(Why, hs[last]) + by # unnormalized log probabilities for next chars
  #Using softmax
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
  for t in reversed(range(len(phrase))):
    dhraw = (1 - (hs[t] * hs[t].T)) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[last]


if __name__ == '__main__':
  data = pickle.load(open('../data.pkl','rb'))

  # Initializing model parameters
  mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  mbh, mby = np.zeros_like(bh), np.zeros_like(by)
  hprev = np.zeros((hidden_size,1))

  #each row has words and then its sentiment
  for row in data:
    if row[1] == "0": #big pos
        target = np.matrix('1;0;0;0;0')
    elif row[1] == "1": #neu
        target = np.matrix('0;1;0;0;0')
    elif row[1] == "2": #neu
        target = np.matrix('0;0;1;0;0')
    elif row[1] == "3": #neg
        target = np.matrix('0;0;0;1;0')
    else: #big neg
        target = np.matrix('0;0;0;0;1')

    seq_length = len(row[0])
    smooth_loss = -np.log(1.0/vector_len)*seq_length # loss at iteration 0

    # forward seq_length characters through the net and fetch gradient
    if row[0]:
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(row[0], target, hprev)
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

  fi = open("vanilla5_model.pkl", "wb")
  pickle.dump(parameter_dict,fi)
  fi.close()
