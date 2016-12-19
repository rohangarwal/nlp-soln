import gensim, pickle, random

if __name__ == "__main__":

	model = gensim.models.Word2Vec.load('w2vmodel')

	'''
	#train data
	filename = '../pickles/train.pkl' #training data pickle file
	data = pickle.load(open(filename,'rb'))
	'''
	
	#test data
	filename = '../pickles/test_SBI.pkl'
	data = pickle.load(open(filename,'rb'))
	
	all_tweet_vec = list()
	
	for line in data:
		tweet_vec = list()
		for word in line[0].split():
			if word in model.vocab:
				tweet_vec.append(model[word])
		all_tweet_vec.append([tweet_vec, line[1]])
		
	with open('tweet_vec_test_SBI.pkl', 'wb') as f:
		pickle.dump(all_tweet_vec, f)
		
	#test data
