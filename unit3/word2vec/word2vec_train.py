import gensim, logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

 
if __name__ == "__main__":

	sentences = MySentences('data/') # a memory-friendly iterator

	#min_count - only words with freq = min_count are kept in vocab while training
	#size - size of the neural net layer of word2vec
	model = gensim.models.Word2Vec(sentences,min_count=5,size=32)

	model.save('data/w2vmodel')
	
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

	'''
	filename = "neg_sentences.txt"
	all_neg_vecs = list()
	with open(filename) as f:
		lines = f.readlines()
		for line in lines:
			review_vec = list()
			for word in line.strip().split():
				if word in model.vocab:
					review_vec.append(model[word])
			all_neg_vecs.append(review_vec)

	with open('neg_vec.pkl', 'wb') as f:
		pickle.dump(all_neg_vecs,f)
	'''