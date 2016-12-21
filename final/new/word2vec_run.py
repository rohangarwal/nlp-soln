import gensim, pickle, random, os

def get_vectors(phrase):
    model = gensim.models.Word2Vec.load(os.path.join(os.path.dirname(__file__)+'/w2vmodel'))
    vec = []
    for word in phrase.split():
        if word in model.vocab:
            vec.append(model[word])
    return vec

if __name__ == "__main__":

	model = gensim.models.Word2Vec.load('w2vmodel')
	
	filename = 'test_phrases.pkl'
	data = pickle.load(open(filename,'rb'))
	all_vec = list()
	count = 0
	found = 0
	for line in data:
		vec = list()
		for word in line[0].split():
			if word in model.vocab:
				vec.append(model[word])
				found += 1
			else :
			    count += 1
			    print word	
		if vec:
			all_vec.append([vec,line[1]])

	print 'Not Found Words - ', count
	print 'Found Words - ', found
	with open('test_phrases_vector.pkl','wb') as f:
		pickle.dump(all_vec,f)
