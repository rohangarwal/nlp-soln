import gensim, pickle, random, os

if __name__ == "__main__":

	model = gensim.models.Word2Vec.load('w2vmodel')

	'''
	#train data
	filename = '../pickles/train.pkl' #training data pickle file
	data = pickle.load(open(filename,'rb'))
	'''
	
	#test data
	dirname = '../pickles'
	dirname2 = '../pickles/files'
	add = 'tweet_vec_'
	for fname in os.listdir(dirname):
		if 'test' in fname:
			#print fname
			data = pickle.load(open(os.path.join(dirname, fname),'rb'))
			all_tweet_vec = list()
			
			for line in data:
				tweet_vec = list()
				for word in line[0].split():
					if word in model.vocab:
						tweet_vec.append(model[word])
				all_tweet_vec.append([tweet_vec, line[1]])
				
			filename = add + fname
			with open(os.path.join(dirname2, filename), 'wb') as f:
				pickle.dump(all_tweet_vec,f)

