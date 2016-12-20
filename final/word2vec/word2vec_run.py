import gensim, pickle, random, os

def get_vectors(phrase):
    model = gensim.models.Word2Vec.load('/home/ksameersrk/Documents/nlp-soln/final/word2vec/w2vmodel')
    vec = []
    for word in phrase.split():
        if word in model.vocab:
            vec.append(model[word])
    return vec

if __name__ == "__main__":

	model = gensim.models.Word2Vec.load('w2vmodel')
	
	filename = '../dataset/pickles/special_train_lines2.pkl'
	data = pickle.load(open(filename,'rb'))
	print len(data)
	all_vec = list()
	for line in data:
		vec = list()
		for word in line[0].split():
			if word in model.vocab:
				vec.append(model[word])
			else :
				print word	
		if vec:
			all_vec.append([vec,line[1]])

	print len(all_vec)
	with open('special_train_lines_vector2.pkl','wb') as f:
		pickle.dump(all_vec,f)
