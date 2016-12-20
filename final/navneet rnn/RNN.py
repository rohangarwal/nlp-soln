#import statements
import cPickle
import copy, numpy as np
np.random.seed(0);

#hyper-parameter for RNN
alpha = 0.001
input_dim = 32
hidden_dim = 32
output_dim = 1

iterations = 6

#different layer for RNN
wt_ih = 2*np.random.random((input_dim,hidden_dim)) - 1
wt_hh = 2*np.random.random((hidden_dim,hidden_dim)) - 1
wt_ho = 2*np.random.random((hidden_dim,output_dim)) - 1

layer_1_values = np.zeros(hidden_dim)

#loading the trained model form the pickle file
print "loading the dump data .... "
model = cPickle.load(open('save1.p', 'rb'))

train_sentence = cPickle.load(open('save2.p', 'rb'))
train_sen_c = cPickle.load(open('save4.p', 'rb'))
train_result = cPickle.load(open('save3.p', 'rb'))

setc = 0;
for i in train_result:
	if(i == 1):
		setc = setc+1;


test_sentence = cPickle.load(open('save5.p', 'rb'))
test_sen_c = cPickle.load(open('save7.p', 'rb'))
test_result = cPickle.load(open('save6.p', 'rb'))

train_vec = []
test_vec = []

#processing the sentence to get 32 bit vector
print "processing dump data ...."
for s1 in train_sentence:
	lx = []
	for w1 in s1:
		temp = model[w1]
		lx.append(temp)
	train_vec.append(lx)

for s1 in test_sentence:
	lx = []
	for w1 in s1:
		temp = model[w1]
		lx.append(temp)
	test_vec.append(lx)
#end of loading and processing .........................



#non linear function and its derivative .................

def sigmoid(x):
	out = 1/(1+np.exp(-x))
	return out

def sigmoid_output_to_derivative(output):
    return output*(1-output)

#........................................................



def forward(l):
	temp1 = np.dot(l,wt_ih)
	temp2 = np.dot(layer_1_values,wt_hh)
	temp3 = temp1+temp2
	layer_1 = sigmoid(temp3)
	temp4 = np.dot(layer_1,wt_ho)
	layer_2 = sigmoid(temp4)
	return temp1,temp2,temp3,layer_1,temp4,layer_2

def test(l):
	global layer_1_values
	layer_1_values = np.zeros(hidden_dim)
	for i in l:
		l1 = np.array(i)
		ret = forward(l1)
		layer_1_values = ret[0]
	print ret[5]
	if(ret[5]>0.5):
		return 1
	else:
		return 0

def error(out,obs):
	return out-obs


def get_dot(l1,l2):
	ret = []

	for i in l1:
		temp = []
		for j in l2:
			temp.append(i*j)
		ret.append(temp)
	return np.array(ret)


def train(l,res,ch,l_lenx,train_s_l):
	global wt_ih
	global wt_hh
	global wt_ho
	global layer_1_values

	if(ch < l_lenx-train_s_l):
		get = forward(l)
		layer_1_values = get[0]
	else:
		get = forward(l)
		layer_1_values = get[0]
		out = float(get[5])
		err = error(out,res)
		del1 = err * sigmoid_output_to_derivative(out)
		delta = alpha * del1 * get[3]
		wt_ho = wt_ho.T - delta
		wt_ho = wt_ho.T		
		scalet_num = 0;
		lx = 0 
		

		for i in wt_ho:
			lx = lx + del1*i[0]
		
		del2 = sigmoid_output_to_derivative(get[3])*lx
		lenc = len(del2)
		ly = []
		lz = []

		r1 = get_dot(del2,l)*alpha
		r2 = get_dot(del2,layer_1_values)*alpha

		wt_ih = wt_ih - r1.T
		wt_hh = wt_hh - r2.T

def backprop(l,res,train_s_l):
	co = 0
	lenx = len(l)-1
	for i in l:
		ch = np.array(i)
		train(ch,res,co,lenx,train_s_l)
		co = co+1



def accuracy(l1,l2):
	count = 0
	len1 = len(l1)
	for i in range(len1):
		print l1[i]," ",l2[i]
		if(l1[i] == l2[i]):
			print "inside if"
			count = count+1 
	res = float(count)/len1
	return res

def train_RNN():
	len2 = len(train_vec)
	for i in range(len2):
		if(i%1000 == 0):
			print i," of ",len2
		backprop(train_vec[i],int(train_result[i]),int(train_sen_c[i]));


def accuracy(l1,l2):
	count = 0
	len1 = len(l1)
	for i in range(len1):
		if(l1[i] == l2[i]):
			count = count+1 
	res = float(count)/len1
	return res


def main():
	for i in range(iterations):
		print "i : ",i
		train_RNN()

	len3 = len(test_vec)

	res1 = []
	res2 = []
	for i in range(len3):
		prd = int(test(test_vec[i]))
		act = int(test_result[i])

		res1.append(prd)
		res2.append(act)

		print "pedicted : ",prd," actual : ",act 

	ret3 = accuracy(res1,res2)*100
	print "The accuracy is : ",ret3,"%"
main()
