import numpy as np
import pickle, sys

# hyperparameters
hidden_size = 16 # size of hidden layer of neurons
learning_rate = 1e-1
vector_len = 50
outputs = 5 #No of dimensions of output

# model parameters
Wxh = np.random.randn(hidden_size, vector_len)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(outputs, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((outputs, 1)) # output bias

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    hs[t] = sigmoid(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state

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
  f_input = sys.argv[1]
  data_old = pickle.load(open(f_input,'rb'))
  f_output = sys.argv[2]
  zero,one,two,three,four = list(),list(),list(),list(),list()
  for i in data_old:
    if str(i[1]) == '0':
      zero.append(i)
    elif str(i[1]) == '1':
      one.append(i)
    elif str(i[1]) == '2':
      two.append(i)
    elif str(i[1]) == '3':
      three.append(i)
    elif str(i[1]) == '4':
      four.append(i)
  min_count = min(len(zero),len(one),len(two),len(three),len(four))
  print 'min_count = ' + str(min_count)
  data_new = list()
  for i in range(0,min_count):
    data_new.append(zero[i])
    data_new.append(one[i])
    data_new.append(two[i])
    data_new.append(three[i])
    data_new.append(four[i])
    
  epochs = 10
  # Initializing model parameters
  mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  mbh, mby = np.zeros_like(bh), np.zeros_like(by)
  
  for epoch in range(0,epochs):
    print 'epoch #:' + str(epoch)
    #each row has words and then its sentiment
    hprev = np.zeros((hidden_size,1))
    for row in data_new:
      if str(row[1]) == "0":
          target = np.matrix('1;0;0;0;0')
      elif str(row[1]) == "1":
          target = np.matrix('0;1;0;0;0')
      elif str(row[1]) == "2":
          target = np.matrix('0;0;1;0;0')
      elif str(row[1]) == "3":
          target = np.matrix('0;0;0;1;0')
      else:
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
    print 'loss = ' + str(loss)          
    


  parameter_dict = {}
  parameter_dict['hprev'] = hprev
  parameter_dict['Why'] = Why
  parameter_dict['by'] = by
  parameter_dict['Wxh'] = Wxh
  parameter_dict['Whh'] = Whh
  parameter_dict['bh'] = bh

  fi = open("models/"+f_output, "wb")
  pickle.dump(parameter_dict,fi)
  fi.close()
