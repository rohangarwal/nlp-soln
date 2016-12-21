import numpy as np
import csv
import sys
import pprint
import dill
import pickle

import parser
from preprocess import clean

from gensim.models import Word2Vec

def relu(z):
	return z * (z > 0)

def relu_prime(z):
	return (z > 0)

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
		
		
		#self.dropout_percent = 0.95

	def forward(self, x, hprev, do_dropout=False):
	
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(hprev)

		seq_length = len(x)
		for t in range(seq_length):
			xs[t] = x[t].reshape(-1, 1)
			hs[t] = self.f(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) 
			#if(do_dropout):
			#	hs[t] *= np.random.binomial(1, self.dropout_percent, size=hs[t-1].shape)
			#else:
			#	hs[t] *= self.dropout_percent
			ys[t] = self.f(np.dot(self.Why, hs[t]) + self.by)
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
			
		# Extra added
		y_pred = []
		for ind in range(seq_length):
			this_y = np.dot(self.Why, hs[ind]) + self.by
			y_pred.append(this_y)
		return np.argmax(y_pred[-1])
		
		#return xs, hs, ys, ps
		
	def forward1(self, x, hprev, do_dropout=False):
	
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(hprev)

		seq_length = len(x)
		
		for t in range(seq_length):
			xs[t] = x[t].reshape(-1, 1)
			hs[t] = self.f(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) 
			#if(do_dropout):
			#	hs[t] *= np.random.binomial(1, self.dropout_percent, size=hs[t-1].shape)
			#else:
			#	hs[t] *= self.dropout_percent
			ys[t] = self.f(np.dot(self.Why, hs[t]) + self.by)
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
			
		
		return xs, hs, ys, ps
		
	def forward2(self, x):
		seq_length = len(x)

		y_pred = []
		dby = np.zeros_like(self.by)
		xs, hs, ys, ps = self.forward1(x, np.zeros((self.hidden_size, 1)))

		for ind in range(seq_length):
			this_y = np.dot(self.Why, hs[ind]) + self.by
			y_pred.append(this_y)
		
		return np.argmax(y_pred[-1])
	

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
			#dhraw = dh * relu_prime(hs[t])
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
			print('Epoch {}'.format(e + 1))

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
					#t = y
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

			print("(val acc: {:.2f}%)".format(self.predict(validation_data) * 100))

		print("\nTraining done.")
		
	def predict(self, testing_data, test=False):
		if testing_data[1] == None:
			predictions = []
			for x in testing_data[0]:
				x = clip(x)
				op = self.forward2(x)
				predictions.append(np.argmax(y))

			return predictions

		else:
			correct = 0
			predictions = {x : 0 for x in range(TYPE)}
			outputs = {x : 0 for x in range(TYPE)}

			l = 0
			for x, y in zip(*testing_data):
				x = clip(x)
				op = self.forward2(x)
				tr = np.argmax(y)
				predictions[op] += 1
				outputs[tr] += 1
				correct = correct + 1 if op == tr else correct + 0
				l += 1

			if test:
				print 'Outputs:\t', outputs
				print 'Predictions:\t', predictions

			return (correct + 0.0) / l


def load_data(filename, count):
	i = 0
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		inputs = []
		outputs = []
		for row in reader:
			inputs.append(row[0])
			outputs.append(int(row[1]))
			i += 1
			if i == count:
				break
		return inputs, outputs

def w2v(sentence):
	words = []
	for word in sentence.split():
		try:
			words.append(model[word])
		except Exception:
			pass

	return np.array(words)

def one_hot(x):
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

def save_model(BRNN):
	with open('brnn_model_%s.pkl' % TYPE, 'wb') as f:
		dill.dump(BRNN, f)

def load_model():
	with open('brnn_model_%s.pkl' % TYPE, 'rb') as f:
		BRNN = dill.load(f)
	return BRNN

if __name__ == "__main__":
	DATA_SIZE = 10000
	TYPE = 3

	INPUT_SIZE = 64
	HIDDEN_SIZE = 16
	OUTPUT_SIZE = TYPE

	#model = Word2Vec.load('model%s' % INPUT_SIZE)

	train_size = DATA_SIZE * 0.8
	val_size = DATA_SIZE * 0.1
	test_size = DATA_SIZE * 0.1
	
	'''
	t_i, t_t =  load_data('train.csv', train_size)
	v_i, v_t = load_data('dev.csv', val_size)
	ts_i, ts_t =  load_data('test.csv', test_size)

	training_inputs = []
	training_targets = []
	for i in range(len(t_i)):
		v = w2v(t_i[i])
		if len(v) == 0:
			continue

		training_inputs.append(v)
		training_targets.append(one_hot(t_t[i]))

	validation_inputs = []
	validation_targets = []
	for i in range(len(v_i)):
		v = w2v(v_i[i])
		if len(v) == 0:
			continue

		validation_inputs.append(v)
		validation_targets.append(one_hot(v_t[i]))

	testing_inputs = []
	testing_targets = []
	for i in range(len(ts_i)):
		v = w2v(ts_i[i])
		if len(v) == 0:
			continue

		testing_inputs.append(v)
		testing_targets.append(one_hot(ts_t[i]))
	'''
    
	train = pickle.load(open('train_phrases_vector.pkl', 'rb'))
	test = pickle.load(open('test_phrases_vector.pkl', 'rb'))

	training_inputs = []
	training_targets = []
	for x in train:
		training_inputs.append(x[0])
		training_targets.append(one_hot(int(x[1])))

	testing_inputs = []
	testing_targets = []
	for x in test:
		testing_inputs.append(x[0])
		testing_targets.append(one_hot(int(x[1])))
	
	print 'Loading data over'
	
	EPOCHS = 10
	LEARNING_RATE = 0.20
	
	TRAIN = True

	BRNN = None
	if TRAIN:
		RNN = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate=LEARNING_RATE)
		RNN.train(training_data=(training_inputs, training_targets), validation_data=(testing_inputs, testing_targets), epochs=EPOCHS, do_dropout=True)
		save_model(RNN)
	else:
		RNN = load_model()

	accuracy = RNN.predict((testing_inputs, testing_targets), True)
	
	print("Accuracy: {:.2f}%".format(accuracy * 100))
	
	
	parameter_dict = {}
	parameter_dict['hprev'] = RNN.hprev
	parameter_dict['Why'] = RNN.Why
	parameter_dict['by'] = RNN.by
	parameter_dict['Wxh'] = RNN.Wxh
	parameter_dict['Whh'] = RNN.Whh
	parameter_dict['bh'] = RNN.bh

	fi = open("new_model", "wb")
	pickle.dump(parameter_dict,fi)
	fi.close()
	
	
	'''
	while True:
		sentence = raw_input("Enter a sentence to parse: ")
		phrases = parser.create_phrases(parser.create_tree(sentence))

		out_p = []
		out_s = []

		for phrase in phrases:
			phrase = clean(phrase)
			v = w2v(phrase)
			if phrase and phrase not in out_p:
				out_p.append(phrase)
				if v.shape[0]:
					out_s.append(BRNN.forward(v))
				else:
					out_s.append(TYPE/2)

		print zip(out_p, out_s)
    '''