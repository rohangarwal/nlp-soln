import numpy as np
import sys
import os
import pprint
import pickle
import dill

def clip(v):
		return v[:10]

class RNN:
	def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
		self.hidden_size = hidden_size 
		self.learning_rate = learning_rate
		self.f = np.tanh
		self.f_prime = lambda x: 1 - (x ** 2)
		self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (hidden_size + input_size))
		self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size * 2))
		self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))
		self.bh = np.zeros((hidden_size, 1)) 
		self.by = np.zeros((output_size, 1))
		self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad	
		self.hprev = None

	def forward(self, x, hprev, do_dropout=False):
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(hprev)
		seq_length = len(x)
		for t in range(seq_length):
			xs[t] = x[t].reshape(-1, 1)
			hs[t] = self.f(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) 
			ys[t] = self.f(np.dot(self.Why, hs[t]) + self.by)
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
			
		y_pred = []
		for ind in range(seq_length):
			this_y = np.dot(self.Why, hs[ind]) + self.by
			y_pred.append(this_y)
		return np.argmax(y_pred[-1])
		
	def forward1(self, x, hprev, do_dropout=False):
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(hprev)
		seq_length = len(x)
		for t in range(seq_length):
			xs[t] = x[t].reshape(-1, 1)
			hs[t] = self.f(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) 
			ys[t] = self.f(np.dot(self.Why, hs[t]) + self.by)
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))		
		
		return xs, hs, ys, ps
		
	def forward2(self, x):
	    return self.forward(x, np.zeros((self.hidden_size, 1)))

	def backprop(self, xs, hs, ys, ps, targets, dy, do_dropout=False):
		dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[-1])

		for t in reversed(range(len(xs))):
			tmp = dy[t] * self.f_prime(ys[t])# * self.dropout_percent
			dWhy += np.dot(tmp, hs[t].T)
			dby += tmp
			dh = np.dot(self.Why.T, dy[t]) + dhnext
			dhraw = dh * (1 - hs[t] ** 2)
			dbh += dhraw
			dWxh += np.dot(dhraw, xs[t].T)
			dWhh += np.dot(dhraw, hs[t-1].T)
			dhnext = np.dot(self.Whh.T, dhraw)

		for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		return dWxh, dWhh, dWhy, dbh, dby, hs[len(xs) - 1]

	def update_params(self, dWxh, dWhh, dWhy, dbh, dby):
		# perform parameter update with Adagrad
		for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
		                            [dWxh, dWhh, dWhy, dbh, dby],
		                            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
			mem += dparam * dparam
			param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
			
			
	def train(self, training_data, validation_data, epochs=5, do_dropout=False):
		for e in range(epochs):
			print('epoch :  {}'.format(e + 1))

			for x, y in zip(*training_data):
				x = clip(x)
				hprev = np.zeros((self.hidden_size, 1))				
				seq_length = len(x)
				xs, hs, ys, ps = self.forward1(x, hprev, do_dropout)
				y_pred = []
				dy = []
				dby = np.zeros_like(self.by)
				for ind in range(seq_length):
					this_y = np.dot(self.Why, hs[ind]) + self.by
					y_pred.append(this_y)

				for ind in range(seq_length):
					this_dy = np.exp(y_pred[ind]) / np.sum(np.exp(y_pred[ind]))
					t = np.argmax(y)
					this_dy[t] -= 1
					dy.append(this_dy)
					dby += this_dy

				y_pred = np.array(y_pred)
				dy = np.array(dy)

				self.mby += dby * dby
				self.by += -self.learning_rate * dby / np.sqrt(self.mby + 1e-8) # adagrad update
				dWxh, dWhh, dWhy, dbh, dby, hprev = self.backprop(xs, hs, ys, ps, y, dy, do_dropout)
				self.hprev = hprev
				self.update_params(dWxh, dWhh, dWhy, dbh, dby)

			print("Accuracy: {:.2f}%".format(self.predict(validation_data) * 100))

		print("Training Completed")
		
	def predict(self, testing_data, test=False):            
		correct = 0
		predictions = {x : 0 for x in range(TYPE)}
		outputs = {x : 0 for x in range(TYPE)}
		l = 0
		TN, TP, FN, FP = 0, 0, 0, 0
		for x, y in zip(*testing_data):
			x = clip(x)
			op = self.forward2(x)
			tr = np.argmax(y)
			predictions[op] += 1
			outputs[tr] += 1
			correct = correct + 1 if op == tr else correct + 0
			l += 1
		if test:
			print 'Targets :\t', outputs
			print 'Predicated :\t', predictions
		return (correct + 0.0) / l



def generate_output_vector(x):
	def three(x):
		if x < 2:
			return 0
		elif x > 2:
			return 2
		else:
			return 1

	v = np.zeros(TYPE)
	if TYPE == 3:
		v[three(x)] = 1
	else:
		v[x] = 1
		
	return v

if __name__ == "__main__":
	TYPE = 5
	INPUT_SIZE = 64
	HIDDEN_SIZE = 16
	OUTPUT_SIZE = TYPE
	EPOCHS = 10
	LEARNING_RATE = 0.20
	TRAIN = False
	
	train = pickle.load(open('train_phrases_vector.pkl', 'rb'))
	test = pickle.load(open('test_phrases_vector.pkl', 'rb'))
	
	training_inputs = []
	training_targets = []
	for x in train:
		training_inputs.append(x[0])
		training_targets.append(generate_output_vector(int(x[1])))

	testing_inputs = []
	testing_targets = []
	for x in test:
		testing_inputs.append(x[0])
		testing_targets.append(generate_output_vector(int(x[1])))

	print 'Loading data over'
	if TRAIN:
		RNN = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate=LEARNING_RATE)
		RNN.train(training_data=(training_inputs, training_targets), validation_data=(testing_inputs, testing_targets), epochs=EPOCHS, do_dropout=True)
		parameter_dict = {}
		parameter_dict['hprev'] = RNN.hprev
		parameter_dict['Why'] = RNN.Why
		parameter_dict['by'] = RNN.by
		parameter_dict['Wxh'] = RNN.Wxh
		parameter_dict['Whh'] = RNN.Whh
		parameter_dict['bh'] = RNN.bh

		fi = open("rnn_model"+str(TYPE)+".pkl", "wb")
		pickle.dump(parameter_dict,fi)
		fi.close()
		
		dill.dump(RNN, open("rnn_object"+str(TYPE), "wb"))
	
	else:
		RNN = dill.load(open('rnn_object'+str(TYPE), 'rb'))

	accuracy = RNN.predict((testing_inputs, testing_targets), True)

	print("Accuracy: {:.2f}%".format(accuracy * 100))
