import gensim, logging, os,pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, filename):
        self.datapkl = pickle.load(open(filename,'rb'))
 
    def __iter__(self):
    	for line in self.datapkl:
    		yield line[0]

if __name__ == "__main__":
	
	filename = '../pickles/train.pkl' #train word2vec
	#sentences = MySentences(filename) # a memory-friendly iterator
	
	sentences = list()
	datapkl = pickle.load(open(filename,'rb'))
	for i in datapkl:
		sentences.append([word for word in i[0].split()])
		#sentences.append(i[0].decode('utf-8'))
	#for i in sentences:
	#	print i
	#min_count - only words with freq = min_count are kept in vocab while training
	#size - size of the neural net layer of word2vec
	
	model = gensim.models.Word2Vec(sentences,min_count=2,size=32)

	model.save('w2vmodel')
	
	#print len(model.vocab.keys())
		
	
	#model = gensim.models.Word2Vec.load('data/w2vmodel')
	
	'''
	filename = "pos_sentences.txt"
	all_pos_vecs = list()
	with open(filename) as f:
		lines = f.readlines()
		for line in lines:
			review_vec = list()
			for word in line.strip().split():
				if word in model.vocab:
					review_vec.append(model[word])
			all_pos_vecs.append(review_vec)

	with open('pos_vec.pkl', 'wb') as f:
		pickle.dump(all_pos_vecs,f)
	
	'''
