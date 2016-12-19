import numpy as np
import pickle

# hyperparameters
hidden_size = 64 # size of hidden layer of neurons
learning_rate = 1e-1
vector_len = 32
outputs = 5 #No of dimensions of output

# model parameters
Wr = np.random.randn(hidden_size, vector_len)*0.01
Wz = np.random.randn(hidden_size, vector_len)*0.01
Wc = np.random.randn(hidden_size, vector_len)*0.01
Ur = np.random.randn(hidden_size, hidden_size)*0.01
Uz = np.random.randn(hidden_size, hidden_size)*0.01
Uc = np.random.randn(hidden_size, hidden_size)*0.01
br = np.zeros((hidden_size, 1)) # hidden bias
bz = np.zeros((hidden_size, 1)) # hidden bias
bc = np.zeros((hidden_size, 1)) # hidden bias
Why = np.random.randn(outputs, hidden_size)*0.01 # hidden to output
by = np.zeros((outputs, 1)) # output bias

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lossFun(phrase, target, hprev):
    """
    phrase,target are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    rs, zs, cs = {}, {}, {}
    rbars, zbars, cbars = {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # forward pass
    # xs represents entire phrase/sentence
    for t in range(len(phrase)):
        xs[t] = np.zeros((vector_len,1)) # encode in 1-of-k representation
        #Copying entire vector for each word

        for j in range(32):
            xs[t][j] = phrase[t][j]

        #GRU Implementation
        rbars[t] = np.dot(Wr, xs[t]) + np.dot(Ur, hs[t-1]) + br
        rs[t] = sigmoid(rbars[t])

        # The z gate, which interpolates between candidate and h[t-1] to compute h[t]
        zbars[t] = np.dot(Wz, xs[t]) + np.dot(Uz, hs[t-1]) + bz
        zs[t] = sigmoid(zbars[t])

        # The candidate, which is computed and used as described above.
        cbars[t] = np.dot(Wc, xs[t]) + np.dot(Uc, np.multiply(rs[t] , hs[t-1])) + bc
        cs[t] = sigmoid(cbars[t])

        ones = np.ones_like(zs[t])
        hs[t] = np.multiply(cs[t],zs[t]) + np.multiply(hs[t-1],ones - zs[t])

    #Many 2 one
    last = len(phrase) - 1  # Getting only last hidden state
    ys = np.dot(Why, hs[last]) + by # unnormalized log probabilities for next chars
    #Using softmax
    ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars

    # compute the vectorized cross-entropy loss
    one = np.ones_like(ps)
    a = np.multiply(target.T , np.log(ps))
    b = np.multiply(one - target.T, np.log(one-ps))
    loss -= (a + b)

    # backward pass: compute gradients going backwards
    dWc = np.zeros_like(Wc)
    dWr = np.zeros_like(Wr)
    dWz = np.zeros_like(Wz)
    dUc = np.zeros_like(Uc)
    dUr = np.zeros_like(Ur)
    dUz = np.zeros_like(Uz)
    dWhy = np.zeros_like(Why)

    # allocate space for the grads of loss with respect to biases
    dbc = np.zeros_like(bc)
    dbr = np.zeros_like(br)
    dbz = np.zeros_like(bz)
    dby = np.zeros_like(by)

    # no error is received from beyond the end of the sequence
    dhnext = np.zeros_like(hs[0])
    drbarnext = np.zeros_like(rbars[0])
    dzbarnext = np.zeros_like(zbars[0])
    dcbarnext = np.zeros_like(cbars[0])
    zs[len(phrase)] = np.zeros_like(zs[0])
    rs[len(phrase)] = np.zeros_like(rs[0])

    dy = np.subtract(ps,target) # backprop into y.
    dWhy += np.dot(dy, hs[last].T)
    dby += dy

    # Not Sure if this was wrong
    #dh = np.dot(Why.T, dy) + dhnext # backprop into

    for t in reversed(xrange(len(phrase))):
        # h[t] influences the cost in 5 ways:

        # through the interpolation using z at t+1
        dha = np.multiply(dhnext, ones - zs[t+1])

        # through transformation by weights into rbar
        dhb = np.dot(Ur.T,drbarnext)

        # through transformation by weights into zbar
        dhc = np.dot(Uz.T,dzbarnext)

        # through transformation by weights into cbar
        dhd = np.multiply(rs[t+1],np.dot(Uc.T,dcbarnext))

        # through the output layer at time t
        dhe = np.dot(Why.T,dy)

        dh = dha + dhb + dhc + dhd + dhe

        dc = np.multiply(dh,zs[t])

        #backprop through tanh
        dcbar = np.multiply(dc , ones - np.square(cs[t]))

        dr = np.multiply(hs[t-1],np.dot(Uc.T,dcbar))
        dz = np.multiply( dh, (cs[t] - hs[t-1]) )

        # backprop through sigmoids
        drbar = np.multiply( dr , np.multiply( rs[t] , (ones - rs[t])) )
        dzbar = np.multiply( dz , np.multiply( zs[t] , (ones - zs[t])) )

        dWr += np.dot(drbar, xs[t].T)
        dWz += np.dot(dzbar, xs[t].T)
        dWc += np.dot(dcbar, xs[t].T)

        dUr += np.dot(drbar, hs[t-1].T)
        dUz += np.dot(dzbar, hs[t-1].T)
        dUc += np.dot(dcbar, np.multiply(rs[t],hs[t-1]).T)

        dbr += drbar
        dbc += dcbar
        dbz += dzbar

        dhnext =    dh
        drbarnext = drbar
        dzbarnext = dzbar
        dcbarnext = dcbar


    '''Clipping Optional
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    '''
    return loss, dWc, dWr, dWz, dUc, dUr, dUz, dWhy, dbc, dbr, dbz, dby, hs[last]


if __name__ == '__main__':
    data = pickle.load(open('../word2vec/train_lines_vector.pkl','rb'))

    # Initializing model parameters
    mWc, mWr, mWz = np.zeros_like(Wc), np.zeros_like(Wr), np.zeros_like(Wz)
    mUc, mUr, mUz = np.zeros_like(Uc), np.zeros_like(Ur), np.zeros_like(Uz)
    mWhy = np.zeros_like(Why)
    mbc, mbr, mbz, mby = np.zeros_like(bc), np.zeros_like(br), np.zeros_like(bz), np.zeros_like(by)
    hprev = np.zeros((hidden_size,1))

    #each row has words and then its sentiment
    for row in data:
        if row[1] == 0:
            target = np.matrix('1;0;0;0;0')
        elif row[1] == 1:
            target = np.matrix('0;1;0;0;0')
        elif row[1] == 2:
            target = np.matrix('0;0;1;0;0')
        elif row[1] == 3:
            target = np.matrix('0;0;0;1;0')
        else:
            target = np.matrix('0;0;0;0;1')

        seq_length = len(row[0])
        smooth_loss = -np.log(1.0/vector_len)*seq_length # loss at iteration 0

        # forward seq_length characters through the net and fetch gradient
        if row[0]:
            loss, dWc, dWr, dWz, dUc, dUr, dUz, dWhy, dbc, dbr, dbz, dby, hprev = lossFun(row[0], target, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([Wc, Wr, Wz, Uc, Ur, Uz, Why, bc, br, bz, by],
                                            [dWc, dWr, dWz, dUc, dUr, dUz, dWhy, dbc, dbr, dbz, dby],
                                            [mWc, mWr, mWz, mUc, mUr, mUz, mWhy, mbc, mbr, mbz, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update


    parameter_dict = {}
    parameter_dict['hprev'] = hprev
    parameter_dict['Why'] = Why
    parameter_dict['Wc'] = Wc
    parameter_dict['Wr'] = Wr
    parameter_dict['Wz'] = Wz
    parameter_dict['Uc'] = Uc
    parameter_dict['Ur'] = Ur
    parameter_dict['Uz'] = Uz
    parameter_dict['by'] = by
    parameter_dict['bc'] = bc
    parameter_dict['br'] = br
    parameter_dict['bz'] = bz

    fi = open("gru3_model.pkl", "wb")
    pickle.dump(parameter_dict,fi)
    fi.close()
