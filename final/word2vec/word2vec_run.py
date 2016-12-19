import gensim, pickle, random, os

if __name__ == "__main__":

	model = gensim.models.Word2Vec.load('w2vmodel')
	
	filename = '../dataset/pickles/test_lines.pkl'
	data = pickle.load(open(filename,'rb'))
	all_vec = list()
	for line in data:
		vec = list()
		for word in line[0].split():
			if word in model.vocab:
				vec.append(model[word])
		if vec:
			all_vec.append([vec,line[1]])

	print len(all_vec)
	with open('test_lines_vector.pkl','wb') as f:
		pickle.dump(all_vec,f)
